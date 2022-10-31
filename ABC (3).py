import random
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def ConvergencePlot(cost_val):
    font = FontProperties();
    font.set_size('larger');
    labels = ["Best cost_val Function", "Mean cost_val Function"]
    plt.figure(figsize=(12.5, 4));
    plt.plot(range(len(cost_val["best"])), cost_val["best"], label=labels[0]);
    plt.scatter(range(len(cost_val["mean"])), cost_val["mean"], color='red', label=labels[1]);
    plt.xlabel("Iteration #");
    plt.ylabel("Value [-]");
    plt.legend(loc="best", prop = font);
    plt.xlim([0,len(cost_val["mean"])]);
    plt.grid();
    plt.show();

def tournament(val_list, sample_size=None):
    score_list = []
    for i in range(len(val_list)):

        if (sample_size != None) and (type(sample_size) is int):
            opponent_entities = random.sample(val_list, sample_size)
        else:
            opponent_entities = val_list

        score_list.append( sum(np.where(val_list[i]>opponent_entities, 1, 0)) )

    return score_list / sum(score_list)

def disruptive(val_list):

    mean_ = np.mean(val_list)
    score_list = []
    for i in range(len(val_list)):
        score_list.append(abs(val_list[i] - mean_))

    return score_list / sum(score_list)

class Bee(object):

    def __init__(self, lower_bound, upper_bound, eval_func, funcon=None):

        self._random(lower_bound, upper_bound)

        if not funcon:
            self.is_valid = True
        else:
            self.is_valid = funcon(self.features)

        if (eval_func != None):
            self.val = eval_func(self.features)
        else:
            self.val = sys.float_info.max
        self._fitness()

        self.ctr = 0

    def _random(self, lower_bound, upper_bound):

        self.features = []
        for i in range(len(lower_bound)):
            self.features.append( lower_bound[i] + random.random() * (upper_bound[i] - lower_bound[i]) )

    def _fitness(self):

        if (self.val >= 0):
            self.fit_val = 1 / (1 + self.val)
        else:
            self.fit_val = 1 + abs(self.val)

class BeeHive(object):

    def run(self):

        cost_val = {}; cost_val["best"] = []; cost_val["mean"] = []
        for iteration in range(self.max_iterations):

            for idx in range(self.size_val):
                self.send_employee(idx)

            self.send_onlookers()

            self.send_scout()

            self.find_best()

            cost_val["best"].append( self.best_val )
            cost_val["mean"].append( sum( [ bee.val for bee in self.pop ] ) / self.size_val )

            if self.verbose:
                self._log(iteration, cost_val)

        return cost_val

    def __init__(self                 ,
                 lower_bound, upper_bound         ,
                 eval_func          = None  ,
                 bee_N    =  30   ,
                 max_iterations     = 100   ,
                 num_max_trials   = None  ,
                 selfunc       = None  ,
                 seed         = None  ,
                 verbose      = False ,
                 extra_parameters = None ,):

        assert (len(upper_bound) == len(lower_bound)), "'lower_bound' and 'upper_bound' must be a list of the same length."

        if (seed == None):
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed
        random.seed(self.seed)

        self.size_val = int((bee_N + bee_N % 2))

        self.dimensions = len(lower_bound)
        self.max_iterations = max_iterations
        if (num_max_trials == None):
            self.num_max_trials = 0.6 * self.size_val * self.dimensions
        else:
            self.num_max_trials = num_max_trials
        self.selfunc = selfunc
        self.extra_parameters = extra_parameters

        self.evaluate = eval_func
        self.lower_bound    = lower_bound
        self.upper_bound    = upper_bound

        self.best_val = sys.float_info.max
        self.sol = None

        self.pop = [ Bee(lower_bound, upper_bound, eval_func) for i in range(self.size_val) ]

        self.find_best()

        self.compute_probability()

        self.verbose = verbose

    def find_best(self):

        val_list = [ bee.val for bee in self.pop ]
        idx  = val_list.index(min(val_list))
        if (val_list[idx] < self.best_val):
            self.best_val     = val_list[idx]
            self.sol = self.pop[idx].features

    def compute_probability(self):

        val_list = [bee.fit_val for bee in self.pop]
        max_values = max(val_list)

        if (self.selfunc == None):
            self.probs = [0.9 * v / max_values + 0.1 for v in val_list]
        else:
            if (self.extra_parameters != None):
                self.probs = self.selfunc(list(val_list), **self.extra_parameters)
            else:
                self.probs = self.selfunc(val_list)

        return [sum(self.probs[:i+1]) for i in range(self.size_val)]

    def send_employee(self, idx):
        single_bee = copy.deepcopy(self.pop[idx])

        dimension_val = random.randint(0, self.dimensions-1)

        bee_idx = idx;
        while (bee_idx == idx): bee_idx = random.randint(0, self.size_val-1)

        single_bee.features[dimension_val] = self._mutate(dimension_val, idx, bee_idx)

        single_bee.features = self._check(single_bee.features, dim=dimension_val)

        single_bee.val = self.evaluate(single_bee.features)
        single_bee._fitness()

        if (single_bee.fit_val > self.pop[idx].fit_val):
            self.pop[idx] = copy.deepcopy(single_bee)
            self.pop[idx].ctr = 0
        else:
            self.pop[idx].ctr += 1

    def send_onlookers(self):
        num_onlookers = 0; beta_val = 0
        while (num_onlookers < self.size_val):

            phi = random.random()

            beta_val += phi * max(self.probs)
            beta_val %= max(self.probs)

            idx = self.select(beta_val)

            self.send_employee(idx)

            num_onlookers += 1

    def select(self, beta_val):
        probs = self.compute_probability()

        for idx in range(self.size_val):
            if (beta_val < probs[idx]):
                return idx

    def send_scout(self):

        all_trials = [ self.pop[i].ctr for i in range(self.size_val) ]

        idx = all_trials.index(max(all_trials))

        if (all_trials[idx] > self.num_max_trials):

            self.pop[idx] = Bee(self.lower_bound, self.upper_bound, self.evaluate)

            self.send_employee(idx)

    def _mutate(self, dim, current_bee, other_bee):

        return self.pop[current_bee].features[dim]    + \
               (random.random() - 0.5) * 2                 * \
               (self.pop[current_bee].features[dim] - self.pop[other_bee].features[dim])

    def _check(self, features, dim=None):


        if (dim == None):
            range_ = range(self.dimensions)
        else:
            range_ = [dim]

        for i in range_:

            if  (features[i] < self.lower_bound[i]):
                features[i] = self.lower_bound[i]

            elif (features[i] > self.upper_bound[i]):
                features[i] = self.upper_bound[i]

        return features

    def _log(self, iteration, cost_val):
        best_val = cost_val["best"][iteration]
        mean_val = cost_val["mean"][iteration]
        msg = f"Iteration {iteration} | Best Eval val = {best_val} | Mean Eval val = {mean_val}"
        print(msg)


def evaluator1(features, a=1, b=100):

    features = np.array(features)

    return (a - features[0])**2 + b * (features[1] - features[0]**2)**2

def evaluator2(features):
    features = np.array(features)

    return 10 * features.size + sum(features*features - 10 * np.cos(2 * np.pi * features))

def run1():
      ndim = int(2)
      model = BeeHive(     lower_bound     =  [0] *ndim ,
                          upper_bound     =  [10]*ndim ,
                          eval_func       = evaluator2 ,
                          bee_N           =         10 ,
                          max_iterations  =         50 )

      cost_val = model.run()

      ConvergencePlot(cost_val)

      print(f"Fitness Value ABC: {model.best_val}")

def run3(bee_Ns):
  colors=['red','yellow','blue','black','green']
  i=0
  plt.figure(figsize=(10, 6))
  for bee_Ni in bee_Ns:
      ndim = int(2)
      model = BeeHive(     lower_bound     =  [0] *ndim ,
                          upper_bound     =  [10]*ndim ,
                          eval_func       = evaluator2 ,
                          bee_N           =         bee_Ni ,
                          max_iterations  =         50 )

      cost_val = model.run()
      plt.plot(range(len(cost_val["best"])), cost_val["best"], label="Highest Fitness, Value Bee_N " + str(bee_Ni ))
      plt.scatter(range(len(cost_val["mean"])), cost_val["mean"],s=10, color=colors[i], label="Average Fitness Value Bee_N " + str(bee_Ni ))
      i=i+1

  plt.legend()
  plt.xlabel("Iteration #");
  plt.ylabel("Fitness Value [-]");
  plt.show()


def run4(bounds):
  plt.figure(figsize=(10, 6))
  colors=['red','yellow','blue','black','green']
  i=0
  for bound in bounds:
      ndim = int(2)
      model = BeeHive(     lower_bound     =  [bound*-1] *ndim ,
                          upper_bound     =  [bound]*ndim ,
                          eval_func       = evaluator2 ,
                          bee_N           =         10 ,
                          max_iterations  =         50 )

      cost_val = model.run()
      plt.plot(range(len(cost_val["best"])), cost_val["best"], label="Highest Fitness, Upper bound " + str([bound]*ndim ) +"Lower bound"+ str([bound*-1] *ndim))
      plt.scatter(range(len(cost_val["mean"])), cost_val["mean"],s=10,color=colors[i], label="Mean Fitness, Upper bound " + str([bound]*ndim ) +"Lower bound"+ str([bound*-1] *ndim));
      i=i+1

  plt.legend()
  plt.xlabel("Iterations");
  plt.ylabel("Fitness Value [-]");
  plt.show()
def run5(iterations):
  plt.figure(figsize=(10, 6))
  colors=['red','yellow','blue','black','green']
  i=0
  for iter in iterations:
      ndim = int(2)
      model = BeeHive(     lower_bound     =  [0] *ndim ,
                          upper_bound     =  [10]*ndim ,
                          eval_func       = evaluator2 ,
                          bee_N           =         10 ,
                          max_iterations  =         iter )

      cost_val = model.run()

      plt.plot(range(len(cost_val["best"])), cost_val["best"], label="Highest Fitness,No of iterations " + str(iter ))
      plt.scatter(range(len(cost_val["mean"])), cost_val["mean"],s=10, label="Average Fitness No of Iterations" + str(iter))
  
  plt.legend()
  plt.xlabel("Iterations");
  plt.ylabel("Fitness Value [-]");
  plt.show()
def run2():
    ndim = int(10)
    model = BeeHive(     lower_bound     = [-5.12]*ndim ,
                         upper_bound     = [ 5.12]*ndim ,
                         eval_func       =   evaluator2 ,
                         bee_N           =           20 ,
                         max_iterations  =           100)


    cost_val = model.run()

    ConvergencePlot(cost_val)



    print(f"Fitness Value ABC: {model.best_val}")

if __name__ == "__main__":
    run2()
    bee_Ns=[20,50,100,150,200]
    iterations= [10,30,50,70,100]
    bounds=[30,50,70,90]
    run5(iterations)
    run4(bounds)
    run3(bee_Ns)