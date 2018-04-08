import math, random

import gym
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import deque
from wrappers import make_atari, wrap_deepmind
from replay_buffer import PrioritizedReplayBuffer

flags = tf.app.flags
flags.DEFINE_boolean("NoisyDQN", True, "if true uses double DQN else uses DQN algorithm")

FLAGS = flags.FLAGS


env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)

class Noise(object) :

    def __init__(self,in_size,out_size,isBias = False) :

        self.in_size = in_size
        self.out_size = out_size
        self.isBias = isBias
        self.randomise()

    def randomise(self) :    
        if not self.isBias :
            self.eps_in = self.f(np.random.random(self.in_size))
            self.eps_out = self.f(np.random.random(self.out_size))
            self.noise = np.dot(np.reshape(self.eps_in,[-1,1]),np.reshape(self.eps_out,[1,-1]))
        else :
            self.noise = self.f(np.random.random(self.out_size))    

    def f(self,x) :
        return np.sign(x) * np.sqrt(np.abs(x))    


class CnnDQN():
    def __init__(self, input_shape, num_actions,scope, std_init=0.4):
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.std_init = std_init
        with tf.variable_scope(scope) as sc :

            self.inp = tf.placeholder(shape = [None]+list(self.input_shape),dtype = tf.float32)
            self.actions = tf.placeholder(shape = [None],dtype = tf.int64)
            self.target = tf.placeholder(shape = [None],dtype = tf.float32)
            self.weights = tf.placeholder(shape = [None],dtype = tf.float32)

            net = self.inp
            net = tf.cast(net/255.0,tf.float32)
            with slim.arg_scope([slim.conv2d],num_outputs = 64,activation_fn = tf.nn.relu) :
                net = slim.conv2d(net,kernel_size = [8,8],stride = [4,4],num_outputs = 32)
                net = slim.conv2d(net,kernel_size = [4,4],stride = [2,2])
                net = slim.conv2d(net,kernel_size = [3,3],stride = [1,1])
            
            net = slim.flatten(net)
            flattened_shape = net.get_shape().as_list()[-1]

            self.noise1_weight  = tf.placeholder(shape = [flattened_shape,512],dtype = tf.float32)
            self.noise1_bias  = tf.placeholder(shape = [512],dtype = tf.float32)
            self.noise2_weight = tf.placeholder(shape = [512,self.num_actions], dtype = tf.float32)
            self.noise2_bias  = tf.placeholder(shape = [self.num_actions],dtype = tf.float32)

            mu1_range = 1 / math.sqrt(flattened_shape)
            self.weight1_mu = tf.get_variable('fc1_mu',shape = [flattened_shape,512],initializer = tf.random_uniform_initializer(-mu1_range, mu1_range))
            self.weight1_sigma = tf.get_variable('fc1_sigma', shape = [flattened_shape, 512], initializer = tf.constant_initializer(self.std_init/math.sqrt(flattened_shape)))
            self.b1_mu = tf.get_variable('b1_mu',shape = [512],initializer = tf.random_uniform_initializer(-mu1_range, mu1_range))
            self.b1_sigma = tf.get_variable('b1_sigma', shape = [512], initializer = tf.constant_initializer(self.std_init/math.sqrt(512)))

            self.noise1_out = tf.nn.relu(tf.matmul(net,self.weight1_mu + tf.multiply(self.weight1_sigma, self.noise1_weight)) + self.b1_mu + tf.multiply(self.b1_sigma,self.noise1_bias))


            mu2_range = 1 / math.sqrt(512)
            self.weight2_mu = tf.get_variable('fc2_mu',shape = [512,self.num_actions],initializer = tf.random_uniform_initializer(-mu2_range, mu2_range))
            self.weight2_sigma = tf.get_variable('fc2_sigma', shape = [512, self.num_actions], initializer = tf.constant_initializer(self.std_init/math.sqrt(512)))
            self.b2_mu = tf.get_variable('b2_mu',shape = [self.num_actions],initializer = tf.random_uniform_initializer(-mu2_range, mu2_range))
            self.b2_sigma = tf.get_variable('b2_sigma', shape = [self.num_actions], initializer = tf.constant_initializer(self.std_init/math.sqrt(self.num_actions)))

            self.noise2_out = tf.matmul(self.noise1_out,self.weight2_mu + tf.multiply(self.weight2_sigma, self.noise2_weight)) + self.b2_mu + tf.multiply(self.b2_sigma,self.noise2_bias)
            self.out = self.noise2_out
            
            action_one_hot = tf.one_hot(self.actions,self.num_actions)
            q_required = tf.reduce_sum(tf.multiply(self.out,action_one_hot),axis = 1)
            self.loss = tf.multiply(tf.square(self.target - q_required),self.weights)
            self.train_step = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(tf.reduce_mean(self.loss)) 
        
        
    def act(self, sess,state):
        state = np.reshape(state,[1] + list(self.input_shape))
        q_value = sess.run(self.out,feed_dict = {self.inp : state, self.noise1_weight : noise1_fc.noise, self.noise1_bias : noise1_bias.noise,
            self.noise2_weight : noise2_fc.noise, self.noise2_bias : noise2_bias.noise})[0]
        action  = np.argmax(q_value)
        
        return action

def update_target_graph(from_scope,to_scope,tau) :
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    ops = []
    for (var1,var2) in zip(from_vars,to_vars) :
        ops.append(var2.assign(var2*tau + (1-tau)*var1))

    return ops

def updateTarget(ops,sess) :
    for op in ops :
        sess.run(op)

model = CnnDQN(env.observation_space.shape, env.action_space.n,'main_model')
target_model = CnnDQN(env.observation_space.shape, env.action_space.n,'target_model')

noise1_fc = Noise(7744,512)
noise1_bias = Noise(0,512, isBias = True)
noise2_fc = Noise(512,env.action_space.n)
noise2_bias = Noise(0,env.action_space.n, isBias = True)

target_noise1_fc = Noise(7744,512)
target_noise1_bias = Noise(0,512, isBias = True)
target_noise2_fc = Noise(512,env.action_space.n)
target_noise2_bias = Noise(0,env.action_space.n, isBias = True)

beta_start = 0.4
beta_frames = 100000
vanillaDQN = False
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

replay_buffer = PrioritizedReplayBuffer(10000, alpha=0.6)

update_ops = update_target_graph('main_model','target_model',tau = 0.0)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

def compute_td_loss_cnn(batch_size,sess, env_shape,beta):
    state, action, reward, next_state, done, weights, indices = replay_buffer.sample(batch_size, beta) 

    state      = np.reshape(np.float32(state),[batch_size] + list(env_shape))
    next_state = np.reshape(np.float32(next_state),[batch_size] + list(env_shape))
    action     = np.reshape(action,[-1])
    reward     = np.reshape(reward,[-1])
    done       = np.reshape(done,[-1])
    weights    = np.reshape(weights,[-1])

    next_q_values = sess.run(target_model.out,feed_dict = {target_model.inp : next_state, target_model.noise1_weight : target_noise1_fc.noise,
        target_model.noise1_bias : target_noise1_bias.noise, target_model.noise2_weight : target_noise2_fc.noise,
        target_model.noise2_bias : target_noise2_bias.noise})

    next_q_value     = np.max(next_q_values,axis = 1)
    
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    _,loss = sess.run([model.train_step, model.loss],feed_dict = {model.inp : state, model.actions : action, model.target : expected_q_value,model.weights : weights,
        model.noise1_weight : noise1_fc.noise, model.noise1_bias : noise1_bias.noise,
        model.noise2_weight : noise2_fc.noise, model.noise2_bias : noise2_bias.noise})
    
    prios = loss + 1e-5
    replay_buffer.update_priorities(indices, prios)
    if FLAGS.NoisyDQN :

        noise1_fc.randomise()
        noise1_bias.randomise()
        noise2_fc.randomise()
        noise2_bias.randomise()
        target_noise1_fc.randomise()
        target_noise1_bias.randomise()
        target_noise2_fc.randomise()
        target_noise2_bias.randomise()   
    return np.mean(loss)


num_frames = 1000000
batch_size = 32
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()
for frame_idx in range(1, num_frames + 1):
    action = model.act(sess,state)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > batch_size:
        beta = beta_by_frame(frame_idx)
        loss = compute_td_loss_cnn(batch_size,sess,env.observation_space.shape,beta)
        losses.append(loss)
        
    if frame_idx % 10000 == 0:
    	print('%d steps reached'%frame_idx)
        print(np.mean(all_rewards[-10:]))
        print(np.mean(losses[-10:]))

    if frame_idx % 1000 == 0 :
        updateTarget(update_ops,sess)

