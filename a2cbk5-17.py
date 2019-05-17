import numpy as np
import tensorflow as tf
#import tensorflow_probability as tfp
#import logz
import os
import time
import inspect
from multiprocessing import Process
import environment
import time
import parameters
import slow_down_cdf
import matplotlib.pyplot as plt

# turn off tensoflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.enable_eager_execution()

# Superparameters, globle parameters
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

pa = parameters.Parameters()
input_width = pa.network_input_width
input_height = pa.network_input_height
output_dim = pa.network_output_dim

# ============================================================================================#
# Utilities
# ============================================================================================#

def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.nn.relu, output_activation=None):
    # raise NotImplementedError
    with tf.variable_scope(scope):
        inph = input_placeholder
        # inph = tf.reshape(inph,[None, 4260])
        # inph = tf.layers.Flatten()(inph)
        inph=tf.expand_dims(inph,axis=1) ### axis could be 0



        conv1 = tf.layers.conv2d(inputs=inph, filters=16, kernel_size=[2, 2], strides=[1, 1], padding='SAME', activation=tf.nn.relu)
        # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[2, 2], strides=[1, 1], padding='SAME',
                                 activation=tf.nn.relu)
        # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        inph = tf.layers.Flatten()(conv2)
        output_placeholder = tf.layers.dense(inph, output_size, activation=output_activation)
    # for ii in range(n_layers):
    #     # inph = tf.layers.dense(inph, size, activation=activation)
    #     inph = tf.layers.dense(inph, 600, activation=activation)
    #     # inph = tf.layers.dense(inputs = inph,units=size,activation = activation,bias_constraint=None)
    # output_placeholder = tf.layers.dense(inph, output_size, activation=output_activation)
    return output_placeholder

def pathlength(path):
    return len(path["reward"])



# ============================================================================================#
# Actor Critic
# ============================================================================================#

class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_advantage_args):
        super(Agent, self).__init__()
        #self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        #self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']
        self.num_target_updates = computation_graph_args['num_target_updates']
        self.num_grad_steps_per_target_update = computation_graph_args['num_grad_steps_per_target_update']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_advantage_args['gamma']
        self.normalize_advantages = estimate_advantage_args['normalize_advantages']

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        tf_config.gpu_options.allow_growth = True  # may need if using GPU
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()  # equivalent to `with self.sess:`
        tf.global_variables_initializer().run()  # pylint: disable=E1101

    def define_placeholders(self):
        """
            Placeholders for batch batch observations / actions / advantages in actor critic
            loss function.
            See Agent.build_computation_graph for notation

            returns:
                sy_ob_no: placeholder for observations
                sy_ac_na: placeholder for actions
                sy_adv_n: placeholder for advantages
        """

        # raise NotImplementedError
        sy_ob_no = tf.placeholder(shape=[None, input_height, input_width], name="ob", dtype=tf.float32)
        # if self.discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        # else:
        #     sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)
        sy_adv_n = tf.placeholder(shape=[None], name="ad", dtype=tf.float32)
        return sy_ob_no, sy_ac_na, sy_adv_n

    def policy_forward_pass(self, sy_ob_no):
        """ Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)

            arguments:
                sy_ob_no: (batch_size, self.ob_dim)

            returns:
                the parameters of the policy.

                if discrete, the parameters are the logits of a categorical distribution
                    over the actions
                    sy_logits_na: (batch_size, self.ac_dim)

                if continuous, the parameters are a tuple (mean, log_std) of a Gaussian
                    distribution over actions. log_std should just be a trainable
                    variable, not a network output.
                    sy_mean: (batch_size, self.ac_dim)
                    sy_logstd: (self.ac_dim,)

            Hint: use the 'build_mlp' function to output the logits (in the discrete case)
                and the mean (in the continuous case).
                Pass in self.n_layers for the 'n_layers' argument, and
                pass in self.size for the 'size' argument.
        """

        sy_logits_na = build_mlp(input_placeholder=sy_ob_no, output_size=self.ac_dim, scope="policy_L",
                                     n_layers=self.n_layers, size=self.size, activation=tf.nn.relu)
        return sy_logits_na

    def sample_action(self, policy_parameters): # policy_params=(None, 6)
        sy_logits_na = policy_parameters

        sy_sampled_ac = tf.multinomial(sy_logits_na, 1)
        sy_sampled_ac = tf.squeeze(sy_sampled_ac, axis=1)
        #print("sy_sampled_ac",tf.Session.run(sy_sampled_ac,{policy_parameters,policy_parameters}))
        return sy_sampled_ac

    def get_log_prob(self, policy_parameters, sy_ac_na): # policy_params=(None, 6)
        sy_logits_na = policy_parameters
        # YOUR_HW2 CODE_HERE
        sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy_ac_na, logits=sy_logits_na)
        return sy_logprob_n

    def build_computation_graph(self):
        """
            Notes on notation:

            Symbolic variables have the prefix sy_, to distinguish them from the numerical values
            that are computed later in the function

            Prefixes and suffixes:
            ob - observation
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)

            Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
            is None

            ----------------------------------------------------------------------------------
            loss: a function of self.sy_logprob_n and self.sy_adv_n that we will differentiate
                to get the policy gradient.
        """
        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n = self.define_placeholders()

        # The policy takes in an observation and produces a distribution over the action space
        self.policy_parameters = self.policy_forward_pass(self.sy_ob_no) # policy_params=(None,6)

        # We can sample actions from this action distribution.
        # This will be called in Agent.sample_trajectory() where we generate a rollout.
        self.sy_sampled_ac = self.sample_action(self.policy_parameters)

        # We can also compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)

        self.actor_loss = tf.reduce_sum(-self.sy_logprob_n * self.sy_adv_n)
        self.actor_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.actor_loss)

        # define the critic
        self.critic_prediction = ((build_mlp(
            self.sy_ob_no,
            1,
            "nn_critic",
            n_layers=self.n_layers,
            size=self.size)))   #tf.squeeze
        self.sy_target_n = tf.placeholder(shape=[None], name="critic_target", dtype=tf.float32)
        self.critic_loss = tf.losses.mean_squared_error(self.sy_target_n, tf.squeeze(self.critic_prediction))#[:,0]
        self.critic_update_op = tf.train.AdamOptimizer(0.001).minimize(self.critic_loss)

    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        env.reset()
        ob = env.observe()
        #print("ob 1 sum",np.sum(ob))
        obs, acs, rewards, next_obs, terminals = [], [], [], [], []
        steps = 0
        while True:
            # if animate_this_episode:
            #     env.render()
            #     time.sleep(0.00001)
            obs.append(ob)
            # raise NotImplementedError
            # print("ob[None].shape",ob[None].shape)
            # input_obs=tf.expand_dims(ob)
            # print("input_obs",input_obs.shape)

            ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: ob[None]})  # YOUR HW2 CODE HERE
            ac = ac[0]
            # if ac == 6:
            #     ac = ac-6

            acs.append(ac)
            ob, rew, done, info = env.step(ac,repeat=True)
            #print("action",ac)
            #print("ob2 sum",np.sum(ob))
            next_obs.append(ob)

            rewards.append(rew)
            steps += 1
            #print("steps",steps)
            # If the episode ended, the corresponding terminal value is 1
            # otherwise, it is 0
            # YOUR CODE HERE
            #print("max_path_length",self.max_path_length)
            if done or steps > self.max_path_length:
                terminals.append(1)
                break
            else:
                terminals.append(0)
        path = {"observation": np.array(obs, dtype=np.float32),
                "reward": np.array(rewards, dtype=np.float32),
                "action": np.array(acs, dtype=np.float32),
                "next_observation": np.array(next_obs, dtype=np.float32),
                "terminal": np.array(terminals, dtype=np.float32),
                "info": info}
        return path

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Estimates the advantage function value for each timestep.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
        """
        # First, estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # To get the advantage, subtract the V(s) to get A(s, a) = Q(s, a) - V(s)
        # This requires calling the critic twice --- to obtain V(s') when calculating Q(s, a),
        # and V(s) when subtracting the baseline
        # Note: don't forget to use terminal_n to cut off the V(s') term when computing Q(s, a)
        # otherwise the values will grow without bound.
        # YOUR CODE HERE
        sum_of_path_lengths = ob_no.shape[0]
        #print("sum_of_path_lengths",sum_of_path_lengths)
        adv_n = []
        for ii in range(sum_of_path_lengths):
            if terminal_n[ii] == 0:
                ob_next = next_ob_no[ii]
                #print("ob_next shape",ob_next.shape)
                ob_current = ob_no[ii]
                #print("[ob_next].shape", [ob_next].shape)
                Vs_next = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: [ob_next]})
                Q_sa = re_n[ii] + self.gamma * Vs_next
                Vs = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: [ob_current]})
                A_sa = Q_sa - Vs
                adv_n.append(A_sa)
            if terminal_n[ii] == 1:
                ob_current = ob_no[ii]
                Q_sa = re_n[ii]
                Vs = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: [ob_current]})
                A_sa = Q_sa - Vs
                adv_n.append(A_sa)

        # raise NotImplementedError
        # adv_n = None

        if self.normalize_advantages:
            # raise NotImplementedError
            adv_n = adv_n = (adv_n - np.mean(adv_n)) / np.std(adv_n)
            # adv_n = None # YOUR_HW2 CODE_HERE
        return adv_n

    def update_critic(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                nothing
        """
        sum_of_path_lengths = ob_no.shape[0]
        n_iteration = self.num_grad_steps_per_target_update * self.num_target_updates
        for i in range(self.num_target_updates):
            target = []
            for ii in range(sum_of_path_lengths):
                if terminal_n[ii] == 0:
                    ob_next = next_ob_no[ii]
                    # ob_current = ob_np[ii]
                    Vs_next = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: [ob_next]})
                    Q_sa = re_n[ii] + self.gamma * Vs_next
                    # Vs = self.sess.run(self.critic_prediction,feed_dict={self.sy_ob_no : ob_current})
                    # A_sa = Q_sa - Vs
                    target.append(Q_sa)
                if terminal_n[ii] == 1:
                    # ob_current = ob_np[ii]
                    # Q_sa = re_n[ii]
                    # Vs = self.sess.run(self.critic_prediction,feed_dict={self.sy_ob_no : ob_current})
                    # A_sa = Q_sa - Vs
                    target.append(re_n[ii])
            for j in range(self.num_grad_steps_per_target_update):
                _ = self.sess.run(self.critic_update_op, feed_dict={self.sy_ob_no: ob_no, self.sy_target_n: target})
        # Use a bootstrapped target values to update the critic
        # Compute the target values r(s, a) + gamma*V(s') by calling the critic to compute V(s')
        # In total, take n=self.num_grad_steps_per_target_update*self.num_target_updates gradient update steps
        # Every self.num_grad_steps_per_target_update steps, recompute the target values
        # by evaluating V(s') on the updated critic
        # Note: don't forget to use terminal_n to cut off the V(s') term when computing the target
        # otherwise the values will grow without bound.
        # YOUR CODE HERE
        # raise NotImplementedError

    def update_actor(self, ob_no, ac_na, adv_n):
        self.sess.run(self.actor_update_op,
                      feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na, self.sy_adv_n: adv_n})

        actor_loss = self.sess.run(self.actor_loss,
                 feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na, self.sy_adv_n: adv_n})
        return actor_loss

def plot_lr_curve(output_file_prefix, max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                  ref_discount_rews, ref_slow_down):
    num_colors = len(ref_discount_rews) + 2
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(mean_rew_lr_curve, linewidth=1.5, label='A2C mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=1.5, label=k)
    ax.plot(max_rew_lr_curve, linewidth=1.5, label='A2C max')

    plt.legend(loc=4)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Discounted Reward", fontsize=16)

    ax = fig.add_subplot(122)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(slow_down_lr_curve, linewidth=1.5, label='A2C mean')
    #print("slow_down_lr_curve",slow_down_lr_curve)
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=1.5, label=k)


    plt.legend(loc=1)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Slowdown", fontsize=16)

    plt.savefig(output_file_prefix + "_lr_curve" + ".pdf")


def process_all_info(trajs):
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in xrange(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in xrange(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in xrange(len(traj['info'].record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)

    return enter_time, finish_time, job_len

def concatenate_all_ob(trajs, pa):

    timesteps_total = 0
    for i in xrange(len(trajs)):
        timesteps_total += len(trajs[i]['reward'])

    all_ob = np.zeros(
        (timesteps_total, 1, pa.network_input_height, pa.network_input_width),
        dtype='float32')

    timesteps = 0
    for i in xrange(len(trajs)):
        for j in xrange(len(trajs[i]['reward'])):
            all_ob[timesteps, 0, :, :] = trajs[i]['ob'][j]
            #print ("all_ob[timesteps, 0, :, :] ", trajs[i]['ob'][j])
            timesteps += 1

    return all_ob


def concatenate_all_ob_across_examples(all_ob, pa):
    num_ex = len(all_ob)
    total_samp = 0
    for i in xrange(num_ex):
        total_samp += all_ob[i].shape[0]

    all_ob_contact = np.zeros(
        (total_samp, 1, pa.network_input_height, pa.network_input_width),
        dtype="float32")

    total_samp = 0

    for i in xrange(num_ex):
        prev_samp = total_samp
        total_samp += all_ob[i].shape[0]
        all_ob_contact[prev_samp : total_samp, :, :, :] = all_ob[i]

    return all_ob_contact

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(xrange(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out



def train_AC(
        exp_name,
        n_iter,
        gamma,
        min_timesteps_per_batch,
        max_path_length,
        learning_rate,
        num_target_updates,
        num_grad_steps_per_target_update,
        animate,
        logdir,
        normalize_advantages,
        seed,
        n_layers,
        size):
    start = time.time()


    # ========================================================================================#
    # Set Up Env
    # ========================================================================================#

    # Make the gym environment
    render = True
    repre = 'image'
    end = 'all_done'
    env = environment.Env(pa, render=render, repre=repre, end=end)



    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    #env.seed(seed)
    # Maximum length for episodes
    max_path_length = max_path_length
    #print("max_path_length",max_path_length)

    #discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    #ob_dim = env.observation_space.shape[0]
    ac_dim = output_dim
    # ========================================================================================#
    # Initialize Agent
    # ========================================================================================#
    computation_graph_args = {
        'n_layers': n_layers,
        #'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        #'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
        'num_target_updates': num_target_updates,
        'num_grad_steps_per_target_update': num_grad_steps_per_target_update,
    }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_advantage_args = {
        'gamma': gamma,
        'normalize_advantages': normalize_advantages,
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_advantage_args)  # estimate_return_args

    # build computation graph
    agent.build_computation_graph()

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    # ========================================================================================#
    # Training Loop
    # ========================================================================================#
    ref_discount_rews, ref_slow_down = slow_down_cdf.launch(pa, pg_resume=None, render=False, plot=False, repre='image',
                                                            end='no_new_job')

    mean_rew_lr_curve = []
    max_rew_lr_curve = []
    slow_down_lr_curve = []



    total_timesteps = 0
    for itr in range(n_iter):
        ###### for plotting
        all_ob = []
        all_action = []
        all_eprews = []
        all_eplens = []
        all_slowdown = []
        ###### for plotting

        print("********** Iteration %i ************" % itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch
        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = np.concatenate([path["reward"] for path in paths])
        next_ob_no = np.concatenate([path["next_observation"] for path in paths])
        terminal_n = np.concatenate([path["terminal"] for path in paths])

        agent.update_critic(ob_no, next_ob_no, re_n, terminal_n)
        adv = agent.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
        # print("adv shape line 560 ",(np.squeeze(adv)).shape)
        # print("ob_no shape",ob_no.shape)
        # print("ac_na shape",ac_na.shape)
        actor_loss = agent.update_actor(ob_no, ac_na, np.squeeze(adv))

        print("actor_loss",actor_loss)

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]

        print("AverageReturn", np.mean(returns))
        #print("EpLenMean", np.mean(ep_lengths))

        enter_time, finish_time, job_len = process_all_info(paths)
        finished_idx = (finish_time >= 0)
        all_slowdown.append((finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx])

        ###### for plotting

        all_ob = ob_no
        all_action = ac_na

        all_eprews.append(
            np.array([discount(path["reward"], pa.discount)[0] for path in paths]))  # episode total rewards
        all_eplens.append(np.array([len(path["reward"]) for path in paths]))  # episode lengths

        eprews = np.concatenate(all_eprews)  # episode total rewards
        eplens = np.concatenate(all_eplens)  # episode lengths
        all_slowdown = np.concatenate(all_slowdown)
        max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
        mean_rew_lr_curve.append(eprews.mean())
        slow_down_lr_curve.append(np.mean(all_slowdown))

        if itr % pa.output_freq == 0:
            plot_lr_curve(pa.output_filename,
                          max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                          ref_discount_rews, ref_slow_down)

        with open(r"data/loss.txt", "ab") as f:
            #      np.savetxt(f, [np.mean(returns)], delimiter = ",")
              f.write('{:.2f}\n'.format(actor_loss))
            #     #f.write("\n")
              f.close()
        ###### for plotting


def main():
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vac')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--n_iter', '-n', type=int, default=2000)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=500)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=4260)
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = 'ac_' + args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = pa.episode_max_length if pa.episode_max_length > 0 else None  # this is set in parameter
    #print("episode_max_length",max_path_length)
    processes = []
    # this is from parameters.py
    pa.compute_dependent_parameters()

    for e in range(args.n_experiments):
        seed = args.seed + 10 * e
        print('Running experiment with seed %d' % seed)

        def train_func():
            train_AC(
                exp_name=args.exp_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                num_target_updates=args.num_target_updates,
                num_grad_steps_per_target_update=args.num_grad_steps_per_target_update,
                animate=args.render,
                logdir=os.path.join(logdir, '%d' % seed),
                normalize_advantages=not (args.dont_normalize_advantages),
                seed=seed,
                n_layers=args.n_layers,
                size=args.size
            )
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_AC in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
