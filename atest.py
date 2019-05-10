import environment
import parameters
import numpy as np

pa = parameters.Parameters()

#pg_resume=None
render=True
repre='image'
end='all_done'

env = environment.Env(pa, render=render, repre=repre, end=end)
env.reset()


while True:
  i = 0
  for i in xrange(5):
     ac = i
     print("action data type:",type(ac))
     ob, rew, done, _ = env.step(ac, repeat=True)
     i = i +1
     if (i > 5):
        i = 0




