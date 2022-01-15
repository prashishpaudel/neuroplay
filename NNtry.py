import random
import numpy as np
from sklearn.preprocessing import normalize
from copy import deepcopy
class population:
    max_layer=5
    crossover_probalility=70
    iteration_count =0
    check_layer = 0
    pop_size = 100
    no_layer=[1 for i in range(pop_size)]
    population_array = []
    time_alive = [0 for _ in range(pop_size)]
    historical_count = 0
    player_x =57
    time_start = [0 for _ in range(pop_size)]
    input_mat = [[0.5, 0.5, 0.5]]
    historical_bird = []
    max_number_neuron = 3
    #copy_nn = []
    #print(max_layer)
    fitness = [1 for i in range(pop_size)]
    #[[[0.2706093998692224], [0.08945316961438265], [0.9702679281592868]]]
    #for non-random [[[0.5123325540633812], [0.23224006997347624], [0.8411383107437302]]]
    for i in range(pop_size):
        temp_3d = [[[(random.uniform(0,1)) ] for j in range(3)]]

        population_array.append(temp_3d)
    temp_population_array = deepcopy(population_array)


    bias = [[random.uniform(0.1,0.7) for j in range(5)] for i in range(pop_size)]
    # print(bias)
    # exit()
    #print(population_array)
    active = 1
    distance_between=[0 for i in range(pop_size)]
    player_y = []
    playerVelY = [-9 for _ in range(pop_size)]  # player's velocity along Y, default same as playerFlapped
    playerMaxVelY = 10  # max vel along Y, max descend speed
    playerMinVelY = -8  # min vel along Y, max ascend speed
    playerAccY = [1 for i in range(pop_size)]  # players downward accleration
    playerFlapAcc = [-9 for i in range(pop_size)]  # players speed on flapping
    playerFlapped = [False for i in range(pop_size)]
    player_height = 0
    # def __init__(self,p):
    #     self.pop_array = p

    def initialize_player_y(self,player_y):
        self.player_y = player_y
    def flap_true(self,i):
        if self.player_y[i] > -2 * self.player_height:
            self.playerVelY[i] = self.playerFlapAcc[i]
            self.playerFlapped[i] = True


    def sigmoid(self,x):
        temp = []
        for i in x[0]:
            temp.append(1 / (1 + np.exp(-i)))
        # print("Yes and i returned ",temp)
        return np.asarray([temp])


    def do_something(self, pipex, pipey,i):
        
        self.input_mat = np.asarray(normalize([[pipex,pipey,self.player_y[i]]]))
        for ii,j in enumerate(self.population_array[i]):
            temp_j = np.asarray(j)
            try:
                result = self.sigmoid(np.dot(self.input_mat,temp_j)+self.bias[i][ii])
            except:
                print("Input_mat ",self.input_mat,"temp_j",temp_j,"Population i,j ",self.population_array[i],"i = ",i,"j = ",ii)
                print("Pipe x ",pipex,"Pipey",pipey,"playery ",self.player_y[i])
                #print(self.input_mat[])
                exit(1000)

            self.input_mat = result

        if result[0]>=.71:
            self.flap_true(i)
    def crossover1(self):
        temp_array = []
        while True:
            a_index = random.randrange(int(len(self.population_array) / 2))
            b_index = random.randrange(int(len(self.population_array) / 2))
            if a_index != b_index:
                break
        x = deepcopy(self.population_array[0])
        y = deepcopy(self.population_array[1])
        long = deepcopy(y)
        short = deepcopy(x)

        if len(long) < len(short):
            temp =deepcopy(long)
            long = deepcopy(short)
            short = deepcopy(temp)
        temp_array = deepcopy(short)
        print("temp_array",temp_array)
        start_index = len(short)
        interface_neuron_no = len(temp_array[len(temp_array)-1][0])
        print(interface_neuron_no," 1?")
        for i in range(start_index,len(long)):
            pass





    def crossover(self):
        temp_array = []
        while True:
            a_index = random.randrange(int(len(self.population_array)/2))
            b_index = random.randrange(int(len(self.population_array)/2))
            if a_index!=b_index:
                break
        x=deepcopy(self.population_array[a_index])
        #x=deepcopy(self.population_array[0])
        y=deepcopy(self.population_array[b_index])
        chance = random.randint(0,30)
        if len(x)==1 and len(y)==1 and chance <=20 :
            temp_A=[[[0] for j in range(3)]]
            temp_A[0][0] = deepcopy(x[0][0])
            temp_A[0][1] = deepcopy(x[0][1])
            temp_A[0][2] = deepcopy(y[0][2])

            for i, j in enumerate(self.population_array):
                if len(j) == 0:
                    self.population_array[i] = deepcopy(temp_A)
                    #print("Crossover with len == at i ",i)
                    break
            #print("len 1 cross dbug null i ", i)
            return

       # print("nope")
        long = deepcopy(y)
        short = deepcopy(x)

        if len(long) < len(short):
            temp = deepcopy(long)
            long = deepcopy(short)
            short = deepcopy(temp)

        for i in range(max(len(x), len(y))):

            if i < min(len(x), len(y)):
                #print("i",i)
                a = np.asarray(deepcopy(x[i]))
                b = np.asarray(deepcopy(y[i]))
                big = deepcopy(a)
                small = deepcopy(b)
                if (a.shape[1] < b.shape[1]):
                    big = deepcopy(b)
                    small = deepcopy(a)
                # print("tmep",temp_array)
                temp_array=deepcopy(short)
                # print("IM HERE")
                # print("TEmp",temp_array)
                # a_temp = [[random.uniform(0.1,0.3) for i in range(small.shape[1])] for j in range(big.shape[1])]
                # a_temp = np.asarray(a_temp).transpose()
                # # print("ATEMP",a_temp)
                # temp_array.append(a_temp.tolist())




            else:
                try:
                    m = np.asarray(temp_array[len(temp_array) - 1]).shape[1]
                except:
                    print("except",temp_array)

                    print("error")
                    exit(555)
                try:
                    add_temp = [[random.uniform(0.1,0.5) for _ in range(len(long[i][0]))] for __ in range(m)]
                except:
                    print("max(len(x), len(y) ",max(len(x), len(y)))
                    print("min(len(x), len(y) ",min(len(x), len(y)))
                    print("I",i)

                    print("Out fo range")
                    exit(4444)
                temp_array.append(add_temp)

        for i,j in enumerate(self.population_array):
                if len(j)==0:
                    if len(temp_array) < self.max_layer:
                        self.population_array[i]=deepcopy(temp_array)
                        # print("Temp ",temp_array)
                        # print("Crossover with topological at  ",i)
                        # print("Child = ", self.population_array[i])
                        # print("parent are ", x, " ", y)

                    else:
                        self.mutate_weight(0,0)
                    break
                #print("crossover dbug null i ", i)





    def mutate_weight(self,t,best_bird_index):

        if t ==1:
            nn_index=best_bird_index
            nn= deepcopy(self.temp_population_array[nn_index])

        else:
            nn_index = random.randint(0, int(self.pop_size / 2))
            nn = deepcopy(self.population_array[nn_index])
        #print("mutate_weight")


        no_of_layer = len(nn)
        try:
            layer_no = random.randrange(0,no_of_layer)
            layer_no1 = random.randrange(0,no_of_layer)
        except:
            # print(no_of_layer)
            # print(nn_index)
            # print(self.population_array[nn_index])
            # print(best_bird_index)
            exit(100)
        x_ = len(nn[layer_no])
        row =random.randrange(0,x_)
        row1 = random.randrange(0,x_)
        y_ = len(nn[layer_no][0])
        column = random.randrange(0,y_)
        column1 = random.randrange(0,y_)
        if row <0 or column<0 or row1<0 or column1<0:
            row1 = 0
            column1=0
            row = 0
            column=0
        if random.randrange(0,2)==1:
            nn[layer_no][row][column] += random.uniform(0.01,0.03)
            nn[layer_no][row1][column1] += random.uniform(0.01,0.03)
        else:
            nn[layer_no][row][column] =random.random()
            #nn[layer_no1][row1][column1] =random.random()
        for i,j in enumerate(self.population_array):
                if len(j)==0:

                    self.population_array[i] = deepcopy(nn)
                    break
        # print("mutate_weight")



    def add_a_layer(self):
        nn_ind = random.randint(0,int(len(self.population_array)/2))
        nn = deepcopy(self.population_array[nn_ind])
        # print(id(nn)," ",id(self.population_array))
        # print(id(nn[0])," ",id(self.population_array[nn_ind][0]))


        # print("adadasdasdasd")
        # exit()
        # print("Before",nn)

        last_layer = nn[len(nn)-1]

        second_last = [[random.uniform(0.1,0.2)]for i in range(len(last_layer))]
        last = [[random.uniform(0.1,0.6)]]

        nn.pop()
        nn.append(second_last)
        nn.append(last)
        if len(nn)>self.max_layer:
            self.mutate_weight(0,0)
            return
        for i,j in enumerate(self.population_array):
            if len(j)==0:
                self.population_array[i]=deepcopy(nn)
                # print("Added a layer at ",i)
                # print("Add gareko parent chahi ",self.population_array[nn_ind])
                self.check_layer = i
                #print(self.population_array[i])
                break

    def add_a_neuron(self,index):
        nn = deepcopy(self.population_array[index])
        temp_nn = deepcopy(nn)
        # print(id(temp_nn[0])," ",id(nn[0])," ",id(self.population_array[index][0]))
        # print(id(temp_nn[1])," ",id(nn[1])," ",id(self.population_array[index][1]))
        # print(id(self.population_array[index])," ",id(nn)," ",temp_nn)
        # print("Adding neuron in ",nn," index ",index)

        for i in range(len(nn)-1):
            if not len(nn[i][0])>self.max_number_neuron:
                temp_neuron = [random.uniform(0.1,0.3) for j in range(len(nn[i]))]
                nn[i]=np.asarray(nn[i])
                nn[i] = nn[i].transpose().tolist()
                nn[i].append(deepcopy(temp_neuron))
                nn[i]=np.asarray(nn[i])
                nn[i]=nn[i].transpose().tolist()
                # print("after addtition before next layer ",nn)
                to_next_layer= [random.uniform(0.1,0.4) for __ in range(len(nn[i+1][0]))]
                nn[i+1].append(to_next_layer)
                for x, y in enumerate(self.population_array):
                    if len(y) == 0:
                        # print("adding this ",nn)
                        self.population_array[index] = deepcopy(temp_nn)
                        print("neuron add gareko chai(parent) with index i ",i," ",self.population_array[index])
                        self.population_array[x] = deepcopy(nn)
                        print(" child ko index ",x," with ",self.population_array[x])
                        #print("added neuron at ",x)
                        exit(2)

                        break
                return


        self.mutate_weight(0,0)



    def add_a_neuron_1(self,index):

        nn = deepcopy(self.population_array[index])
        copy_nn =deepcopy(self.population_array[index])





        for i,j in enumerate(nn):
            if not len(j[0])>self.max_number_neuron:
                #print(nn[i])
                neuron = [random.uniform(0.1,0.5) for i in range(len(j))]
                nn[i] = np.asarray(nn[i])
                nn[i] = nn[i].transpose()
                nn[i]=nn[i].tolist()
                nn[i].append(neuron)

                nn[i] = np.asarray(nn[i])
                nn[i] = nn[i].transpose()

                nn[i]=nn[i].tolist()
                # print("NNi+1",nn[i+1])
                # temporary_nn.append(np.copy(np.asarray(j)).tolist())
                # print(nn[i])
                to_next_layer = [random.random() for _ in range(len(nn[i+1][0]))]

                nn[i+1].append(deepcopy(to_next_layer))

                for x, y in enumerate(self.population_array):
                    if len(y) == 0:
                        self.population_array[x] = deepcopy(nn)

                        self.population_array[index] = deepcopy(copy_nn)

                        exit()
                        return

        self.mutate_weight(0,0)


    def genetic(self):
        total_fitness = 0
        best_bird_index=0
        for i,j in enumerate(self.time_alive):
            if j == max(self.time_alive):
                best_bird_index = i
        self.fitness = [self.time_alive[i]**15/(self.distance_between[i]+1) for i in range(self.pop_size)]


        choosed_birds = []

        total_fitness = sum(self.fitness)
        x =100/total_fitness

        for i in range(self.pop_size):
            self.fitness[i] = self.fitness[i]* x
        for i in range(int(self.pop_size/2)-1):
            accumulate_fitness = self.fitness[0]
            t = random.randrange(100)

            for j in range(self.pop_size):
                if t <= accumulate_fitness:
                        choosed_birds.append(j)
                        self.population_array[i] = deepcopy(self.temp_population_array[j])
                        self.bias[i] = deepcopy(self.bias[i])

                        break
                accumulate_fitness += self.fitness[j+1]
       # print("Choosed birds",choosed_birds)
       #  for x,y in enumerate(self.population_array):
       #      if x<len(choosed_birds):
       #          #print(self.temp_population_array[choosed_birds[x]],"at i ",x)
       #          pass
        # print("Before 3 added")
        # print(self.population_array)
        # print("Best bird",self.population_array[int(self.pop_size/2)-1])
        # if self.iteration_count>9:
        #     self.population_array[84]=[[[0.2897291310388391], [0.16226794505194986], [0.9729753744197761]]]
        # self.iteration_count+=1

            #print(j," at i ",i)
        self.historical_bird.append(self.temp_population_array[best_bird_index])
        self.historical_count += 1
        for index in range(len(self.historical_bird)):
            self.population_array[int(self.pop_size/2)-1+index]=self.historical_bird[index]
        #self.population_array[0] = [[[0.15100948254060553], [0.0453052758592859], [0.794261430311989]]]
        if self.historical_count == 3:
            self.historical_bird.pop(0)
            self.historical_count-=1
        self.mutate_weight(1,best_bird_index)
        self.mutate_weight(1,best_bird_index)

        # print("Before all crossovers and mutations ")
        # for i,j in enumerate(self.population_array):
        #     print(j,"at i ",i)

        for i in range(int(self.pop_size/2)+1+len(self.historical_bird), self.pop_size):
            if random.randrange(100)<self.crossover_probalility:
                self.crossover()

            else:
                a=random.randrange(100)
                if a < 85:
                    self.mutate_weight(0, 0)
                elif 85<a<94:
                    add_neuron_flag = 0
                    for i in range(int(self.pop_size)):
                        test_index = random.randrange(int(self.pop_size/2))
                        if not len(self.population_array[test_index])<2:
                            self.add_a_neuron(test_index)
                            add_neuron_flag+=1
                            break
                    if add_neuron_flag!=1:
                        self.mutate_weight(0,0)

                else:
                    #print("add\n\n\n\n\n\n")
                    self.add_a_layer()


        # for i,j in enumerate(self.temp_population_array):
        #     if len(j)>1:
        #         print(j,"at i ",i)
        for i in range(int(self.pop_size/2),self.pop_size):
            for j in range(self.max_layer):
                self.bias[i][j] = random.uniform(0.1,0.4)

        self.temp_population_array = deepcopy(self.population_array)
        # print("After all ")








        # print("before",self.temp_population_array)
        # if self.active == 1 and self.iteration_count==0:
        #     for j in range(3):
        #         if j==1:
        #             self.temp_3d[0][j] = [random.uniform(0.019,0.020236)]
        #         elif j==2:
        #             self.temp_3d[0][j] = [random.uniform(0.92,0.96)]
        #         else:
        #             self.temp_3d[0][j] = [random.uniform(0.15,0.1608430)]
        #     self.temp_population_array = np.copy(np.asarray([self.temp_3d for i in range(self.pop_size)])).tolist()
        #     #print(self.temp_population_array)
        #     # exit()
        #     print("actually IN")
        # self.iteration_count+=1
        # print("after", self.temp_population_array)
        # for i,j in enumerate(self.population_array):
        #     self.no_layer[i]=len(j)







    #def do_something(self):

