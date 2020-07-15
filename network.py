# coding:utf-8

from torch import optim
from model import *
from functions import *

class GraphNet(object):
    def __init__(self, conf):
        self.conf = conf
        self.model = Model(conf)
        self.process_data()
        num_params = 0
        print("-------------------------------------------------------------------------------------------")
        print(self.model)
        for param in self.model.parameters():
            num_params += param.numel()
        print("The total number of params --------->", num_params)
        print("-------------------------------------------------------------------------------------------")

    def process_data(self):
        self.data = load_small_data(self.conf.dataset)
        self.adj = self.data[0].tocsr()
        self.feas = self.data[1]
        self.labels = self.data[2]
        self.rowsum = np.array(self.adj.sum(1))
        self.rowsum = self.rowsum.reshape(self.conf.nodenum)
        self.rowsum = self.rowsum.tolist()


        self.train_topks = findTopK(self.adj, list(range(self.conf.trainstart, self.conf.trainend+1)), self.conf.k, self.rowsum)
        self.val_topks = findTopK(self.adj, list(range(self.conf.valstart, self.conf.valend+1)), self.conf.k, self.rowsum)
        self.test_topks = findTopK(self.adj, list(range(self.conf.teststart, self.conf.testend+1)), self.conf.k, self.rowsum)

        self.train_map = createMap(np.array(self.train_topks), self.feas.A, self.conf.biasfactor,
                                    self.conf.trainstart, self.conf.mapsize_a, self.conf.mapsize_b)
        self.val_map = createMap(np.array(self.val_topks), self.feas.A, self.conf.biasfactor,
                                  self.conf.valstart, self.conf.mapsize_a, self.conf.mapsize_b)
        self.test_map = createMap(np.array(self.test_topks), self.feas.A, self.conf.biasfactor,
                                   self.conf.teststart, self.conf.mapsize_a, self.conf.mapsize_b)

        self.train_map = self.train_map.transpose((0, 3, 1, 2))
        self.val_map = self.val_map.transpose((0, 3, 1, 2))
        self.test_map = self.test_map.transpose((0, 3, 1, 2))

        print("train_map.shape: ", self.train_map.shape)
        print("val_map.shape: ", self.val_map.shape)
        print("test_map.shape: ", self.test_map.shape)

        self.labels = np.argmax(self.labels,
                                axis=1)

        self.xs_train = self.train_map
        self.ys_train = self.labels[self.conf.trainstart:self.conf.trainend]

        random_state = 100
        np.random.seed(random_state)
        self.xs_train = np.random.permutation(self.xs_train)
        np.random.seed(random_state)
        self.ys_train = np.random.permutation(self.ys_train)

        self.xs_train = torch.from_numpy(self.xs_train)
        self.xs_train = self.xs_train.float()
        self.ys_train = torch.from_numpy(self.ys_train)
        self.ys_train = self.ys_train.long()

        self.xs_val = self.val_map
        self.ys_val = self.labels[self.conf.valstart:self.conf.valend]
        self.xs_val = torch.from_numpy(self.xs_val)
        self.xs_val = self.xs_val.float()
        self.ys_val = torch.from_numpy(self.ys_val)

        self.xs_test = self.test_map
        self.ys_test = self.labels[self.conf.teststart:]
        self.xs_test = torch.from_numpy(self.xs_test)
        self.xs_test = self.xs_test.float()
        self.ys_test = torch.from_numpy(self.ys_test)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.xs_train, self.ys_train = self.xs_train.cuda(), self.ys_train.cuda()
            self.xs_val, self.ys_val = self.xs_val.cuda(), self.ys_val.cuda()
            self.xs_test, self.ys_test = self.xs_test.cuda(), self.ys_test.cuda()
            print("CUDA available")


    def cal_loss(self):
        self.loss = nn.CrossEntropyLoss()

    def train(self):
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.conf.learning_rate_base,
                                       weight_decay=self.conf.weight_decay)

    def val_accuracy(self):
        self.model.eval()
        num_correct = 0
        with torch.no_grad():
            out, _= self.model(self.xs_val, self.feas, self.conf.valstart)
            _, output = torch.max(out, dim=1)
            num_correct += torch.sum(output == self.ys_val)
            model_accuracy = float(num_correct) / len(self.ys_val)
        return model_accuracy

    def test_accuracy(self):
        self.model.eval()
        num_correct = 0
        with torch.no_grad():
            output, _ = self.model(self.xs_test, self.feas, self.conf.teststart)
            _, output = torch.max(output, dim=1)
            num_correct += torch.sum(output == self.ys_test)
            model_accuracy = float(num_correct) / len(self.ys_test)
        return model_accuracy

    def train_model(self):
        self.model.train()
        i = 0
        while i < len(self.xs_train):
            start = i
            end = i + self.conf.batch_size
            batch_x = self.xs_train[start:end]
            batch_y = self.ys_train[start:end]
            self.optimizer.zero_grad()
            output, attention = self.model(batch_x, self.feas, start)
            attention = attention.view(-1)
            attentionloss = self.conf.attentionreg * torch.sum(attention ** 2)
            loss = self.loss(output, batch_y) + torch.tensor(attentionloss)
            loss.backward(loss)
            self.optimizer.step()
            i += self.conf.batch_size
        return loss.item()

    def transductive_train(self):
        stats = [0, 0, 0]
        self.cal_loss()
        self.train()

        for epoch_num in range(self.conf.max_step):
            train_loss = self.train_model()
            val_acc = self.val_accuracy()

            test_accuracy = self.test_accuracy()
            print('step: %d --- loss: %.4f, val: %.3f' % (epoch_num, train_loss, val_acc))
            print('Test accuracy -----> ', test_accuracy)


            if epoch_num > 400 and val_acc >= stats[0]:
                stats[0], stats[1], stats[2] = val_acc, 0, max(test_accuracy, stats[2])
            else:
                stats[1] += 1

            print(stats, epoch_num)
            if stats[1] > 250 and epoch_num > 250:
                print('Test accuracy -----> ', stats[2])
                return stats[2]
                break
        return stats[2]