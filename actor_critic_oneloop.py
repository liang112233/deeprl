"""
Using:
tensorflow 1.0
Sisheng Liang  2019
"""

import numpy as np
import tensorflow as tf
import environment
import time
import parameters
import slow_down_cdf
import matplotlib.pyplot as plt
from multiprocessing import Process


# Superparameters, globle parameters
OUTPUT_GRAPH = True  # If this is False, no graph in tensorboard
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

pa = parameters.Parameters()
input_width = pa.network_input_width
input_height = pa.network_input_height
output_dim = pa.network_output_dim


class Actor(object):
    def __init__(self, sess,  input_height, input_width,lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, shape=[1,input_height, input_width], name="state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=input_width,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=input_width,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l2,
                units=output_dim,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        #s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions   this is an array

        csprob_n = np.cumsum(probs)
        act = (csprob_n > np.random.rand()).argmax()

        #print("action probality", act_prob,"action", act)

        return act
    def get_prob(self,s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions   this is an array

            # print("action probality", act_prob,"action", act)

        return probs



class Critic(object):
    def __init__(self, sess, input_height, input_width,lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, shape=[1,input_height, input_width], name="state")
        self.v_ = tf.placeholder(tf.float32, None, "v_next")
        self.r = tf.placeholder(tf.float32, None, name='r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=input_width,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            l2 = tf.layers.dense(
                inputs=l1,
                units=input_width,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )
            v_out = tf.layers.dense(
                inputs=l2,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )
            self.v = tf.reduce_sum(v_out)

        with tf.variable_scope('squared_TD_error'):

            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s = s[np.newaxis, :]
        s_ = s_[np.newaxis, :]
        v_ = self.sess.run(self.v, feed_dict={self.s: s_})   # value @ next state
        #v = self.sess.run(self.v, feed_dict={self.s: s})  # value @ next state
        #print("value next, value",v_,v)
        #print('v.shape:', self.v.get_shape(), self.s.get_shape(), self.v_)
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r}) #,self.v: v})
        return td_error


sess = tf.Session()
actor = Actor(sess,input_height, input_width, lr=LR_A)
critic = Critic(sess,input_height, input_width,lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor
sess.run(tf.global_variables_initializer())
# #
if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)




def plot_lr_curve(output_file_prefix, max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                  ref_discount_rews, ref_slow_down):
    num_colors = len(ref_discount_rews) + 2
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(mean_rew_lr_curve, linewidth=2, label='PG mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)
    ax.plot(max_rew_lr_curve, linewidth=2, label='PG max')

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(122)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(slow_down_lr_curve, linewidth=2, label='PG mean')
    print("slow_down_lr_curve",slow_down_lr_curve)
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=2, label=k)


    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Slowdown", fontsize=20)

    plt.savefig(output_file_prefix + "_lr_curve" + ".pdf")








def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):

    env = environment.Env(pa, render=render, repre=repre, end=end)
    #print "env",env.get_reward()

    # ----------------------------
    print("Preparing for data...")
    for iteration in xrange(pa.num_epochs):

            # Collect trajectories until we get timesteps_per_batch total timesteps
        trajs = []
        env.reset()
        obs = []
        acts = []
        rews = []
        info = []
        ob = env.observe()
         #ob = ob[np.newaxis, :]
         #print("ob shape", np.shape(ob))

         #print("ob in get traj",ob)

        for _ in xrange(pa.episode_max_length):
             act_prob = actor.get_prob(ob)
             print("act_prob",act_prob)

             #print("act")
             csprob_n = np.cumsum(act_prob)
             a = (csprob_n > np.random.rand()).argmax()

             obs.append(ob)  # store the ob at current decision making step
             acts.append(a)

             ob_next, rew, done, info = env.step(a, repeat=True)

             td_error = critic.learn(ob, rew, ob_next)  # gradient = grad[r + gamma * V(s_) - V(s)]
             actor.learn(ob, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]


             #print("ob",ob)
             print("rews",rews)
             print ("rew",rew)

             rews.append(rew)
             ob = ob_next

             #print("act_prob",act_prob)
             #print("get_entropy(act_prob)",get_entropy(act_prob))

             if done: break
             if render: env.render()




def main():

    import parameters
    pa = parameters.Parameters()
    np.random.seed(2)
    tf.set_random_seed(2)  # reproducible

    pa.compute_dependent_parameters()

    pg_resume = None
    # pg_resume = 'data/tmp_0.pkl'

    render = False
    # #
    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    launch(pa, pg_resume, render, repre='image', end='all_done')


if __name__ == '__main__':
    main()









############
# this is for reference plotting


env = environment.Env(pa, render=False, repre='image', end='no_new_job')
env.reset()
ref_discount_rews, ref_slow_down = slow_down_cdf.launch(pa, pg_resume=None, render=False, plot=False, repre='image', end='no_new_job')

mean_rew_lr_curve = []
max_rew_lr_curve = []
slow_down_lr_curve = []
timer_start = time.time()

for iteration in xrange(pa.num_epochs):
    # obs = []
    # acts = []
    # rews = []
    # entropy = []
    # info = []
    # track_r = []
    all_ob = []
    all_action = []
    all_eprews = []
    all_eplens = []
    all_slowdown = []


    # s = env.observe() # ob is state
    # s = s[np.newaxis, :]
    # print("s shape",s.shape)

    # go through all examples
    for ex in xrange(pa.num_ex):

        # Collect trajectories until we get timesteps_per_batch total timesteps
        trajs = []

        for i in xrange(pa.num_seq_per_batch):
            traj = get_traj(actor, env, pa.episode_max_length)
            # print("traj",traj)
            trajs.append(traj)

        # roll to next example
        env.seq_no = (env.seq_no + 1) % env.pa.num_ex
        all_ob.append(concatenate_all_ob(trajs, pa))
        # print "rt",traj
        rets = [discount(traj["reward"], pa.discount) for traj in trajs]
        all_action.append(np.concatenate([traj["action"] for traj in trajs]))
        all_eprews.append(
            np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs]))  # episode total rewards
        all_eplens.append(np.array([len(traj["reward"]) for traj in trajs]))  # episode lengths



        # All Job Stat
        enter_time, finish_time, job_len = process_all_info(trajs)
        finished_idx = (finish_time >= 0)
        all_slowdown.append((finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx])

    all_ob = concatenate_all_ob_across_examples(all_ob, pa)
    all_action = np.concatenate(all_action)

    all_eprews.append(np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs]))  # episode total rewards
    all_eplens.append(np.array([len(traj["reward"]) for traj in trajs]))  # episode lengths

    eprews = np.concatenate(all_eprews)  # episode total rewards
    eplens = np.concatenate(all_eplens)  # episode lengths
    all_slowdown = np.concatenate(all_slowdown)




    max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
    mean_rew_lr_curve.append(eprews.mean())
    slow_down_lr_curve.append(np.mean(all_slowdown))
    print("max_rew_lr_curve",max_rew_lr_curve)
    print("mean_rew_lr_curve",mean_rew_lr_curve)
    print("slow_down_lr_curve",slow_down_lr_curve)

    if iteration % pa.output_freq == 0:
        param_file = open(pa.output_filename + '_' + str(iteration) + '.pkl', 'wb')
        #cPickle.dump(pg_learner.get_params(), param_file, -1)
        param_file.close()

        #slow_down_cdf.launch(pa, pa.output_filename + '_' + str(iteration) + '.pkl', render=False, plot=True, repre='image', end='no_new_job')

        plot_lr_curve(pa.output_filename,
                      max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                      ref_discount_rews, ref_slow_down)

    # act_prob = actor.get_prob(s)    #this should get from my actor network output
    # csprob_n = np.cumsum(act_prob)
    # a = (csprob_n > np.random.rand()).argmax()
    # s_, r, done, info = env.step(a,repeat=True)
    # s_ = s_[np.newaxis, :]  # s @t and s @ t+1


    #
    # #print("s_",s_)
    # print("action ",a)
    # print("r",r)
    # print("s_",s_)
    # print("ob shape", s_.shape)
    # #
    # #if done: r = -20
#
    # track_r.append(r)
    # td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
    # actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]
    # s = s_
