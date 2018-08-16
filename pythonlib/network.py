import os
import time
import numpy as np
import tensorflow as tf

from . import dataset

class Network():
    def __init__(self,config):
        self.config = config
        if self.config["input_size"] is not None:
            self.h,self.w = self.config.get("input_size",(25,25))
        else:
            self.h,self.w = None,None
        self.category_num = self.config.get("category_num",21)
        self.accum_num = self.config.get("accum_num",1)
        self.data = self.config.get("data",None)
        #self.sess = self.config.get("sess",tf.Session())
        self.net = {}
        self.weights = {}
        self.trainable_list = []
        self.loss = {}
        self.images = {}
        self.metrics = {}
        self.summary = {"train":{"op":None},"val":{"op":None}}
        self.saver = {}
        self.print_detail_count_limit = 2

        self.variables={"total":[]}
        self.l2loss = {"total":0}

    def build(self,net_input,net_label):
        if "output" not in self.net:
            with tf.name_scope("placeholder"):
                self.net["input"] = net_input
                self.net["label"] = net_label

            self.net["output"] = self.create_network()
            self.pred()
        return self.net["output"]

    # need to rewrite
    def create_network(self,layer):
        if "init_model_path" in self.config:
            self.load_init_model()
        return layer # note no softmax

    # need to rewrite
    def pred(self):
        if self.h is not None:
            self.net["rescale_output"] = tf.image.resize_bilinear(self.net["output"],(self.h,self.w))
        else:
            label_size = tf.py_func(lambda x:x.shape[1:3],[self.net["input"]],[tf.int64,tf.int64])
            self.net["rescale_output"] = tf.image.resize_bilinear(self.net["output"],[tf.cast(label_size[0],tf.int32),tf.cast(label_size[1],tf.int32)])
            
        self.net["pred"] = tf.argmax(self.net["rescale_output"],axis=3)

    # need to rewrite
    def load_init_model(self):
        model_path = self.config["init_model_path"]
        self.init_model = np.load(model_path,encoding="latin1").item()
        print("load init model success: %s" % model_path)

    # need to rewrite
    def get_weights_and_biases(self,layer):
        if layer in self.weights:
            return self.weights[layer]
        w,b = None,None
        self.weights[layer] = (w,b)
        self.trainable_list.append(w)
        self.trainable_list.append(b)
        self.variables["total"].append(w)
        self.variables["total"].append(b)
        return w,b
    
    # need to rewrite
    def getloss(self):
        label,output = self.remove_ignore_label(tf.cast(self.net["label"],tf.int32),self.net["rescale_output"])
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=output))
        return loss

    def loss_summary(self,weight_decay):
            self.loss["norm"] = self.getloss()
            self.l2loss["total"] = sum([tf.nn.l2_loss(self.weights[layer][0]) for layer in self.weights])
            self.loss["l2"] = self.l2loss["total"]
            self.loss["total"] = self.loss["norm"] + weight_decay*self.l2loss["total"]
            return ["total","l2","norm"]

    # need to rewrite
    def optimize(self,base_lr,momentum):
        self.net["lr"] = tf.Variable(base_lr, trainable=False)
        #opt = tf.train.AdamOptimizer(self.net["lr"])
        opt = tf.train.MomentumOptimizer(self.net["lr"],momentum)
        g0,ag0,accum0,update0,clean0,train_op0 = self.optimize_single(opt,self.loss["total"])

        self.grad = {}
        for g in [g0]:
            for key in g:
                self.grad[key] = g[key]

        self.net["accum_gradient"] = []
        for ag in [ag0]:
            self.net["accum_gradient"].extend(ag)

        self.net["accum_gradient_accum"] = []
        for accum in [accum0]:
            self.net["accum_gradient_accum"].extend(accum)
        
        with tf.control_dependencies([update0]):
            self.net["accum_gradient_update"] = tf.no_op()

        self.net["accum_gradient_clean"] = []
        for clean in [clean0]:
            self.net["accum_gradient_clean"].extend(clean)

    def optimize_single(self,opt,loss,variable_list=None):
        if variable_list is None:
            gradients = opt.compute_gradients(loss)
        else:
            gradients = opt.compute_gradients(loss,var_list=variable_list)
        grad = {}
        accum_gradient = []
        accum_gradient_accum = []
        new_gradients = []
        old_gradients = []
        for (g,v) in gradients:
            if g is None: continue
            if v in self.lr_2_list:
                g = 2*g
            if v in self.lr_10_list:
                g = 10*g
            if v in self.lr_20_list:
                g = 20*g
            #if v in self.lr_100_list:
            #    g = 100*g
            b = g/(v+1e-20)
            grad[v.name] = {}
            grad[v.name]["grad"] = g
            grad[v.name]["weight"] = v
            grad[v.name]["rate"] = b
            accum_gradient.append(tf.Variable(tf.zeros_like(g),trainable=False))
            accum_gradient_accum.append(accum_gradient[-1].assign_add( g/self.accum_num, use_locking=True))
            new_gradients.append((accum_gradient[-1],v))
            old_gradients.append((g,v))

        accum_gradient_clean = [g.assign(tf.zeros_like(g)) for g in accum_gradient]
        accum_gradient_update  = opt.apply_gradients(new_gradients)
        train_op = opt.apply_gradients(old_gradients)

        return grad,accum_gradient,accum_gradient_accum,accum_gradient_update,accum_gradient_clean,train_op

    # need to rewrite 86lines
    def train(self,base_lr,weight_decay,momentum,batch_size,epoches):
        assert self.data is not None,"data is None"
        assert self.sess is not None,"sess is None"
        self.net["is_training"] = tf.placeholder(tf.bool)

        x_train,y_train,id_train,iterator_train = self.data.next_batch(category="train",batch_size=batch_size,epoches=-1)
        x_val,y_val,id_val,iterator_val = self.data.next_batch(category="val",batch_size=batch_size,epoches=-1)
        x = tf.cond(self.net["is_training"],lambda:x_train,lambda:x_val)
        y = tf.cond(self.net["is_training"],lambda:y_train,lambda:y_val)
        self.build()
        self.pre_train(base_lr,weight_decay,momentum,batch_size,save_layers=["input","rescale_output","output","label","pred","is_training"])

        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(iterator_train.initializer)
            self.sess.run(iterator_val.initializer)

            if self.config.get("model_path",False) is not False:
                print("start to load model: %s" % self.config.get("model_path"))
                print("before l2 loss:%f" % self.sess.run(self.loss["l2"]))
                self.restore_from_model(self.saver["norm"],self.config.get("model_path"),checkpoint=False)
                print("model loaded ...")
                print("after l2 loss:%f" % self.sess.run(self.loss["l2"]))

            start_time = time.time()
            print("start_time: %f" % start_time)
            print("config -- lr:%f weight_decay:%f momentum:%f batch_size:%f epoches:%f" % (base_lr,weight_decay,momentum,batch_size,epoches))

            epoch,i = 0.0,0
            iterations_per_epoch_train = self.data.get_data_len() // batch_size
            self.metrics["best_val_miou"] = 0.6
            while epoch < epoches:
                if i == 0: # to protect restore
                    self.sess.run(tf.assign(self.net["lr"],base_lr))
                if i == 40*iterations_per_epoch_train:
                    new_lr = 0.003
                    print("save model before new_lr:%f" % new_lr)
                    self.saver["lr"].save(self.sess,os.path.join(self.config.get("saver_path","saver"),"lr-%f" % base_lr),global_step=i)
                    self.sess.run(tf.assign(self.net["lr"],new_lr))

                data_x,data_y = self.sess.run([x,y],feed_dict={self.net["is_training"]:True})
                params = {self.net["input"]:data_x,self.net["label"]:data_y}
                self.sess.run(self.net["accum_gradient_accum"],feed_dict=params)
                if i % self.accum_num == self.accum_num - 1:
                    _ = self.sess.run(self.net["accum_gradient_update"])
                    _ = self.sess.run(self.net["accum_gradient_clean"])

                if i%100 in [0,1,2,3,4,5,6,7,8,9]:
                    self.sess.run(self.metrics["update"],feed_dict=params)
                if i%100 == 9:
                    summarys,accu,miou,loss,lr = self.sess.run([self.summary["train"]["op"],self.metrics["accu"],self.metrics["miou"],self.loss["total"],self.net["lr"]],feed_dict=params)
                    self.summary["writer"].add_summary(summarys,i)
                    print("epoch:%f, iteration:%f, lr:%f, loss:%f, accu:%f, miou:%f" % (epoch,i,lr,loss,accu,miou))
                if i%100 == 10:
                    self.sess.run(self.metrics["reset"],feed_dict=params)

                if i%1000 in [10,11,12,13,14,15,16,17,18,19]:
                    data_x,data_y = self.sess.run([x,y],feed_dict={self.net["is_training"]:False})
                    params = {self.net["input"]:data_x,self.net["label"]:data_y}
                    self.sess.run(self.metrics["update"],feed_dict=params)
                if i%1000 == 19:
                    data_x,data_y = self.sess.run([x,y],feed_dict={self.net["is_training"]:False})
                    params = {self.net["input"]:data_x,self.net["label"]:data_y}
                    summarys,accu,miou,loss,lr = self.sess.run([self.summary["val"]["op"],self.metrics["accu"],self.metrics["miou"],self.loss["total"],self.net["lr"]],feed_dict=params)
                    self.summary["writer"].add_summary(summarys,i)
                    if miou > self.metrics["best_val_miou"]:
                        self.saver["best"].save(self.sess,os.path.join(self.config.get("saver_path","saver"),"best-val-miou-%f" % miou),global_step=i)
                        self.metrics["best_val_miou"] = miou
                    print("val epoch:%f, iteration:%f, lr:%f, loss:%f, accu:%f, miou:%f" % (epoch,i,lr,loss,accu,miou))
                if i%1000 == 20:
                    data_x,data_y = self.sess.run([x,y],feed_dict={self.net["is_training"]:False})
                    params = {self.net["input"]:data_x,self.net["label"]:data_y}
                    self.sess.run(self.metrics["reset"],feed_dict=params)

                if i%3000 == 2999:
                    self.saver["norm"].save(self.sess,os.path.join(self.config.get("saver_path","saver"),"norm"),global_step=i)
                i+=1
                epoch = i / iterations_per_epoch_train

            end_time = time.time()
            print("end_time:%f" % end_time)
            print("duration time:%f" %  (end_time-start_time))

    def remove_ignore_label(self,gt,output=None,pred=None): 
        ''' 
        gt: not one-hot 
        output: a distriution of all labels, and is scaled to macth the size of gt
        NOTE the result is a flatted tensor
        and all label which is bigger that or equal to self.category_num is void label
        '''
        gt = tf.reshape(gt,shape=[-1])
        indices = tf.squeeze(tf.where(tf.less(gt,self.category_num)),axis=1)
        gt = tf.gather(gt,indices)
        if output is not None:
            output = tf.reshape(output, shape=[-1,self.category_num])
            output = tf.gather(output,indices)
            return gt,output
        elif pred is not None:
            pred = tf.reshape(pred, shape=[-1])
            pred = tf.gather(pred,indices)
            return gt,pred

    def pre_train(self,base_lr,weight_decay,momentum,batch_size,save_layers=["input","output","label","pred"]):
        self.loss_summary(weight_decay)
        self.optimize(base_lr,momentum)
        for layer in save_layers:
            tf.add_to_collection(layer,self.net[layer])

        self.saver["norm"] = tf.train.Saver(max_to_keep=2,var_list=self.trainable_list)
        self.saver["lr"] = tf.train.Saver(var_list=self.trainable_list)
        self.saver["best"] = tf.train.Saver(var_list=self.trainable_list,max_to_keep=2)

    def restore_from_model(self,saver,model_path,checkpoint=False):
        assert self.sess is not None
        if checkpoint is True:
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            saver.restore(self.sess, model_path)
