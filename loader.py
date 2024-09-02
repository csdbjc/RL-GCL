import functools
import logging
import os
import warnings
import time
import random
import joblib
import torch
import pickle
import json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataListLoader
from itertools import repeat
from copy import deepcopy
from baselines.common import schedules
from baselines.deepq import replay_buffer
import tensorflow as tf

import deep_q_networks
from cite.aug import drop_nodes, permute_edges, subgraph, mask_nodes
from cite.result import Molecule
from enviroment import MultiObjectiveRewardMolecule, smile_list2graph_list
from mol_function import get_similarity, AVALON_fingerprint
from cite.Agent import Agent

warnings.filterwarnings("ignore", category=UserWarning)


class SortedQueue:
    def __init__(self, sort_predicate=None):
        self.sort_predicate = sort_predicate
        self.data_ = []

    def contains(self, smiles):
        return any(d['id'] == smiles for d in self.data_)

    def insert(self, data):
        assert 'id' in data
        assert 'reward' in data

        if self.contains(data['id']):
            return

        self.data_.append(data)
        self.data_.sort(key=self.sort_predicate, reverse=True)
        self.data_ = self.data_[:1]  # choose the transformed molecule with the highest reward

    def extend(self, queue):
        assert isinstance(queue, SortedQueue)

        for data in queue.data_:
            self.insert(data)

class Environment(Molecule):
    def __init__(
            self,
            graph_pred,
            original_molecule,
            device,
            **kwargs
    ):
        super(Environment, self).__init__(**kwargs)

        self.original_molecule = original_molecule.to(device)
        self.graph_pred = graph_pred
        self.loss = np.Inf
        self.gamma = 0.9
        self.device = device

    def _reward(self):
        self.original_molecule = self.original_molecule.to(self.device)
        if len(self.original_molecule.smiles) == 1:
            self.original_molecule.smiles = self.original_molecule.smiles[0]
        transform_molecule = smile_list2graph_list([self._state], self.original_molecule.y.cpu())[0].to(self.device)
        self.graph_pred.eval()
        self.graph_pred.predict_f = True

        is_labeled = self.original_molecule.y == self.original_molecule.y
        is_labeled = is_labeled.float()
        task = torch.sum(is_labeled)
        original_label = self.graph_pred([self.original_molecule]) * is_labeled
        transformed_label = self.graph_pred([transform_molecule]) * is_labeled
        diff_score = torch.sum(abs(original_label - transformed_label)) / task
        sim_score = get_similarity(self.original_molecule, transform_molecule)
        reward = 0.5 * diff_score + 0.5 * sim_score

        return {
            'pyg': transform_molecule,
            'reward': float(reward),
            'diff_score': float(diff_score),
            'sim_score': float(sim_score)
        }

        # transform_mol = Chem.MolFromSmiles(self._state)
        # qed_score = QED.qed(transform_mol)
        # sim_score = get_similarity(self.init_mol, self._state)
        # diff_score = abs(qed_score - QED.qed(Chem.MolFromSmiles(self.init_mol)))
        # reward = (sim_score + diff_score) * self.gamma ** (self.max_steps - self._counter)
        # return {
        #     'smiles': transform_mol,
        #     'reward': float(reward),
        #     'qed_score': float(qed_score),
        #     'diff_score': float(diff_score),
        #     'sim_score': float(sim_score)
        # }


def meg_train(action_encoder,
              n_input,
              environment,
              queue,
              device,
              max_steps_per_episode,
              epochs=3,
              batch_sizes=4):
    agent = Agent(n_input, int(n_input / 2), 1, device, 1e-4, 200)

    eps = 0.9
    batch_losses = []
    episode = 0
    it = 0

    while episode < epochs:
        steps_left = max_steps_per_episode - environment.num_steps_taken  # remaining steps
        valid_actions = list(environment.get_valid_actions())  # get valid actions
        observations = np.vstack(
            [
                np.append(action_encoder(action), steps_left)  # state is (m, t), m is a valid molecule, t is the remaining steps
                for action in valid_actions
            ]
        )
        observations = torch.as_tensor(observations).float()  # action taken by the agent when it encounters state
        a = agent.action_step(observations, eps)
        action = valid_actions[a]  # transformed molecule
        result = environment.step(action)
        _, out, done = result

        action_embedding = np.append(
            action_encoder(action),
            steps_left
        )

        action_embeddings = np.vstack(
            [
                np.append(action_encoder(action), steps_left)
                for action in list(environment.get_valid_actions())
            ]
        )

        agent.replay_buffer.push(
            torch.as_tensor(action_embedding).float(),
            torch.as_tensor(out['reward']).float(),
            torch.as_tensor(action_embeddings).float(),
            float(result.terminated)
        )

        if it % 1 == 0 and len(agent.replay_buffer) >= batch_sizes:
            loss = agent.train_step(
                batch_sizes,
                environment.gamma,
                0.995
            )
            loss = loss.item()
            batch_losses.append(loss)

        it += 1

        if done:
            episode += 1

            print(f'Episode {episode}> Reward = {out["reward"]:.4f}  diff_score = {out["qed_score"]:.4f}  sim_score = {out["sim_score"]:.4f}')
            queue.insert({
                'id': action,
                **out
            })
            # torch.save(environment.model.state_dict(), './pre-trained/model_pretrain.pth')
            eps *= 0.9
            environment.initialize()
            # eps = max(eps, 0.05)

def _step(environment, dqn, memory, episode, hparams, exploration, head):
  """Runs a single step within an episode.

  Args:
    environment: molecules.Molecule; the environment to run on.
    dqn: DeepQNetwork used for estimating rewards.
    memory: ReplayBuffer used to store observations and rewards.
    episode: Integer episode number.
    hparams: HParams.
    exploration: Schedule used for exploration in the environment.
    head: Integer index of the DeepQNetwork head to use.

  Returns:
    molecules.Result object containing the result of the step.
  """
  # Compute the encoding for each valid action from the current state.
  steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
  valid_actions = list(environment.get_valid_actions())
  observations = np.vstack([
      np.append(deep_q_networks.get_fingerprint(act, hparams), steps_left)
      for act in valid_actions
  ])
  action = valid_actions[dqn.get_action(
      observations, head=head, update_epsilon=exploration.value(episode))]
  result = environment.step(action)
  action_fingerprints = np.vstack([
      np.append(deep_q_networks.get_fingerprint(act, hparams), steps_left)
      for act in environment.get_valid_actions()
  ])
  # we store the fingerprint of the action in obs_t so action
  # does not matter here.
  memory.add(
      obs_t=np.append(
          deep_q_networks.get_fingerprint(action, hparams), steps_left),
      action=0,
      reward=result.reward,
      obs_tp1=action_fingerprints,
      done=float(result.terminated))
  return result


def _episode(environment, dqn, memory, episode, global_step, hparams, exploration, beta_schedule, queue):
  """Runs a single episode.

  Args:
    environment: molecules.Molecule; the environment to run on.
    dqn: DeepQNetwork used for estimating rewards.
    memory: ReplayBuffer used to store observations and rewards.
    episode: Integer episode number.
    global_step: Integer global step; the total number of steps across all
      episodes.
    hparams: HParams.
    summary_writer: FileWriter used for writing Summary protos.
    exploration: Schedule used for exploration in the environment.
    beta_schedule: Schedule used for prioritized replay buffers.

  Returns:
    Updated global_step.
  """
  episode_start_time = time.time()
  environment.initialize()
  if hparams.num_bootstrap_heads:
    head = np.random.randint(hparams.num_bootstrap_heads)
  else:
    head = 0
  for step in range(hparams.max_steps_per_episode):
    result = _step(
        environment=environment,
        dqn=dqn,
        memory=memory,
        episode=episode,
        hparams=hparams,
        exploration=exploration,
        head=head)
    flag = False
    if step == hparams.max_steps_per_episode - 1:
      # episode_summary = dqn.log_result(result.state, result.reward)
      print('Episode {:d} reward:{:.4f} (1-sim):{:.4f} (1-diff):{:.4f}'.format(
          episode + 1, np.dot(result.reward, dqn.objective_weight)[0], result.reward[0], result.reward[1]))
      if np.dot(result.reward, dqn.objective_weight)[0] > 0.6:
          flag = True
      queue.insert({
          'id': result.state,
          'pyg': smile_list2graph_list([result.state], environment.target_pyg.y.cpu())[0],
          'reward': np.dot(result.reward, dqn.objective_weight)[0],
          'transform_score': result.transform_score
      })
      logging.info('Episode %d/%d took %gs', episode + 1, hparams.num_episodes,
                   time.time() - episode_start_time)
      logging.info('SMILES: %s\n', result.state)
      # Use %s since reward can be a tuple or a float number.
      logging.info('The reward is: %s', str(result.reward))
    if (episode > min(50, hparams.num_episodes / 10)) and (
        global_step % hparams.learning_frequency == 0):
      if hparams.prioritized:
        (state_t, _, reward_t, state_tp1, done_mask, weight,
         indices) = memory.sample(
             hparams.batch_size, beta=beta_schedule.value(episode))
      else:
        (state_t, _, reward_t, state_tp1,
         done_mask) = memory.sample(hparams.batch_size)
        weight = np.ones([reward_t.shape[0]])
      # np.atleast_2d cannot be used here because a new dimension will
      # be always added in the front and there is no way of changing this.
      if reward_t.ndim == 1:
        reward_t = np.expand_dims(reward_t, axis=1)
      td_error, error_summary, _ = dqn.train(
          states=state_t,
          rewards=reward_t,
          next_states=state_tp1,
          done=np.expand_dims(done_mask, axis=1),
          weight=np.expand_dims(weight, axis=1))
      logging.info('Current TD error: %.4f', np.mean(np.abs(td_error)))
      if hparams.prioritized:
        memory.update_priorities(
            indices,
            np.abs(np.squeeze(td_error) + hparams.prioritized_epsilon).tolist())
    global_step += 1
  return global_step, flag

def run_training(hparams, environment, dqn, queue):
  """Runs the training procedure.

  Briefly, the agent runs the action network to get an action to take in
  the environment. The state transition and reward are stored in the memory.
  Periodically the agent samples a batch of samples from the memory to
  update(train) its Q network. Note that the Q network and the action network
  share the same set of parameters, so the action network is also updated by
  the samples of (state, action, next_state, reward) batches.


  Args:
    hparams: tf.HParams. The hyper parameters of the model.
    environment: molecules.Molecule. The environment to run on.
    dqn: An instance of the DeepQNetwork class.

  Returns:
    None
  """
  num_episodes = 10
  tf.reset_default_graph()
  with tf.Session() as sess:
    dqn.build()
    # model_saver = tf.train.Saver(max_to_keep=hparams.max_num_checkpoints)
    # The schedule for the epsilon in epsilon greedy policy.
    exploration = schedules.PiecewiseSchedule(
        [(0, 1.0), (int(hparams.num_episodes / 2), 0.1),
         (hparams.num_episodes, 0.01)],
        outside_value=0.01)
    if hparams.prioritized:
      memory = replay_buffer.PrioritizedReplayBuffer(hparams.replay_buffer_size,
                                                     hparams.prioritized_alpha)
      beta_schedule = schedules.LinearSchedule(
          hparams.num_episodes, initial_p=hparams.prioritized_beta, final_p=0)
    else:
      memory = replay_buffer.ReplayBuffer(hparams.replay_buffer_size)
      beta_schedule = None
    sess.run(tf.global_variables_initializer())
    sess.run(dqn.update_op)
    global_step = 0
    for episode in range(num_episodes):
        # 随机权重
      # sim_weight = random.random()
      # dqn.objective_weight = np.array([[sim_weight], [1 - sim_weight]])
      # logging.info('Episode {:d} ObjWeight {}'.format(episode, str(dqn.objective_weight)))
      global_step, done = _episode(
          environment=environment,
          dqn=dqn,
          memory=memory,
          episode=episode,
          global_step=global_step,
          hparams=hparams,
          exploration=exploration,
          beta_schedule=beta_schedule,
          queue=queue)
      if done:
          break
      if (episode + 1) % hparams.update_frequency == 0:
        sess.run(dqn.update_op)

def hard_pair(dataset, original_pyg):  # generate hard positive and negative sample pairs
    hparams = deep_q_networks.get_hparams()  # 导入参数
    hparams.atom_types = dataset.atoms_  # 数据集的原子集合
    environment = MultiObjectiveRewardMolecule(
        target_pyg=original_pyg,
        device=dataset.device,
        graph_pred=dataset.graph_pred,
        atom_types=set(hparams.atom_types),
        init_mol=original_pyg.smiles[0],
        allow_removal=hparams.allow_removal,
        allow_no_modification=hparams.allow_no_modification,
        allow_bonds_between_rings=hparams.allow_bonds_between_rings,
        allowed_ring_sizes=hparams.allowed_ring_sizes,
        max_steps=hparams.max_steps_per_episode)
    dqn = deep_q_networks.MultiObjectiveDeepQNetwork(
        objective_weight=np.array([[0.3], [0.7]]),
        input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
        q_fn=functools.partial(deep_q_networks.multi_layer_model, hparams=hparams),
        optimizer=hparams.optimizer,
        grad_clipping=hparams.grad_clipping,
        num_bootstrap_heads=hparams.num_bootstrap_heads,
        gamma=hparams.gamma,
        epsilon=0.8)

    queue = SortedQueue(sort_predicate=lambda mol: mol['reward'])
    if dataset.dataset_name in ['tox21', 'hiv', 'pcba', 'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox', 'mutag']:
        environment.task_type = 'cls'
    else:
        environment.task_type = 'reg'

    run_training(
        hparams=hparams,
        environment=environment,
        dqn=dqn,
        queue=queue
    )
    return queue


    # queue = SortedQueue(sort_predicate=lambda mol: mol['reward'])  # stored generated molecule
    # params = {
    #     # General-purpose params
    #     'init_mol': original_molecule.smiles[0],
    #     'atom_types': set(dataset.atoms_),
    #     'max_steps': dataset.max_step,
    #     # Task-specific params
    #     'original_molecule': original_molecule,
    #     'device': torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu'),
    #     'graph_pred': dataset.graph_pred
    # }
    # env = Environment(**params)
    # env.initialize()
    #
    # def action_encoder(action):  # AVALON
    #     fp = AVALON_fingerprint(action)
    #     return fp
    #
    # meg_train(action_encoder,
    #           512 + 1,
    #           env,
    #           queue,
    #           device=torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu'),
    #           max_steps_per_episode=dataset.max_step)
    # return queue

def save_results(base_path, queue, n):  # save hard molecule
    picture_dir = base_path + "/pictures"
    json_dir = base_path + "/jsons"
    # pyg_dir = base_path + "/pyg"
    s = str(n)

    if not os.path.exists(picture_dir):
        os.makedirs(picture_dir)
        os.makedirs(json_dir)
        # os.makedirs(pyg_dir)

    for i, molecule in enumerate(queue):
        pyg = deepcopy(molecule.pop('pyg'))
        # with open(pyg_dir + "/graph_{}_{}".format(s, str(i)), 'wb') as pf:
        #     pickle.dump(pyg, pf)
        if isinstance(pyg.smiles, list):
            smiles = pyg.smiles[0]
        else:
            smiles = pyg.smiles
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=(450, 300))
        img.save(picture_dir + "/picture_{}_{}".format(s, str(i)) + '.png')
    with open(json_dir + "/molecule_{}.json".format(s), "w") as outf:
        json.dump(queue, outf, indent=2)

def reinforcement(dataset, data, idx):
    temp = deepcopy(data)
    aug_graph, aug_reward = temp, 0.
    if os.path.exists('reinforcement/{}/jsons/molecule_{}.json'.format(dataset.dataset_name, str(idx))):
        # f_graph = open('reinforcement/{}/pyg/graph_{}_1'.format(dataset.dataset_name, str(idx)), 'rb')
        # aug_graph = pickle.load(f_graph)
        f_json = open('reinforcement/{}/jsons/molecule_{}.json'.format(dataset.dataset_name, str(idx)), 'r')
        json_file = json.loads(f_json.read())
        aug_reward = json_file[1]['reward']
        aug_smiles = json_file[1]['id']
        aug_y = json_file[1]['transform_score']
        aug_graph = smile_list2graph_list([aug_smiles], [aug_y])[0]
    else:
        if data.edge_index.shape[1] > 5:  # If it is [Se] a single element or the element has less than 3 edges, it is not generated
            aug_queue = hard_pair(dataset, data)
            aug_graph = aug_queue.data_[0]['pyg']
            aug_reward = aug_queue.data_[0]['reward']
            # dataset.neg_df.loc[int(idx), 'smiles'] = neg_queue.data_[0]['id']
            # dataset.neg_df.loc[int(idx), 'qed'] = neg_queue.data_[0]['qed_score']
            # dataset.neg_df.to_csv('data/neg_{}.csv'.format(dataset.dataset_name))
            overall_queue = [{
                'pyg': data,
                'smiles': data.smiles[0],
                'original_label': data.y[0].tolist()
            }]
            overall_queue.extend(aug_queue.data_)
            save_results('./reinforcement/{}'.format(dataset.dataset_name), overall_queue, idx)

    return aug_graph, aug_reward


class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, device, max_step=3, transform=None, pre_transform=None, dataset_name=None, aug='none'):
        self.dataset_name = dataset_name
        self.get_atoms()
        self.max_step = max_step
        self.aug = aug
        self.device = device
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        if os.path.exists(self.processed_dir + '//atom.json'):
            with open(self.processed_dir + '//atom.json', 'r') as f:
                self.atoms_ = json.load(f)
                f.close()
        else:
            self.atoms_ = []

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def get_atoms(self):  # gets all the constituent atoms of the data set
        atoms_ = []
        smiles_list = self.smiles()
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            for atom in mol.GetAtoms():
                c = atom.GetSymbol()
                if c not in atoms_:
                    atoms_.append(c)
        return atoms_

    def smiles(self):
        df = pd.read_csv('data/{}.csv'.format(self.dataset_name))
        smiles_list = df['smiles'].values.tolist()
        smiles_list = [smiles.replace('\n', '') for smiles in smiles_list]
        for t, smiles in enumerate(smiles_list.copy()):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:  # does not comply with valency rules
                smiles_list.remove(smiles)
                continue
        return smiles_list

    def target(self):
        df = pd.read_csv('data/{}.csv'.format(self.dataset_name))
        y = df.iloc[:, 1::].values.tolist()
        smiles_list = df['smiles'].values.tolist()
        for t, smiles in enumerate(smiles_list.copy()):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:  # does not comply with valency rules
                smiles_list.remove(smiles)
                y.pop(t)
                continue
        y = np.array(y)
        return y

    def process(self):
        self.atoms_ = self.get_atoms()
        with open(self.processed_dir + '/atom.json', 'w') as f:
            json.dump(self.atoms_, f)
        smiles_list = self.smiles()
        y = self.target()
        data_list = smile_list2graph_list(smiles_list, y)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):  # In DataLoader, the get() function is called to get each sample and its label
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        if self.aug == 'none':
            data = data
        elif self.aug == 'reinforcement':
            data, reward = reinforcement(self, data, idx)
            data.batch = torch.zeros(data.x.shape[0]).long()
            return data.cpu(), reward
        elif self.aug == 'random':
            if os.path.exists('random/{}/pyg/graph_{}_1'.format(self.dataset_name, str(idx))):
                f = open('random/{}/pyg/graph_{}_1'.format(self.dataset_name, str(idx)), 'rb')
                data = pickle.load(f)
            else:
                n = np.random.randint(3)
                if n == 0:
                    data = drop_nodes(deepcopy(data))
                elif n == 1:
                    data = permute_edges(deepcopy(data))
                elif n == 2:
                    data = subgraph(deepcopy(data))
                else:
                    print('sample error')
                    assert False
                data.batch = torch.zeros(data.x.shape[0]).long()
                # base_path = './random/{}'.format(self.dataset_name)
                # picture_dir = base_path + "/picture"
                # json_dir = base_path + "/jsons"
                # pyg_dir = base_path + "/pyg"
                # if not os.path.exists(picture_dir):
                #     os.makedirs(picture_dir)
                #     os.makedirs(json_dir)
                #     os.makedirs(pyg_dir)
                # with open(pyg_dir + "/graph_{}_1".format(str(idx)), 'wb') as pf:
                #     pickle.dump(data, pf)
            return data.cpu()
        else:
            print('augmnentation error')
        return data
