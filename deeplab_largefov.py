import os
import sys
import time
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

from pythonlib.network import Network
from pythonlib.crf import crf_inference
from pythonlib.dataset import dataset

class DeepLab_largefov(Network):
    def __init__(self,config):
        self.min_prob = 0.0001
        super(DeepLab_largefov,self).__init__(config)
        self.stride = {}
        self.stride["input"] = 1

        # different lr for different variable
        self.lr_1_list = []
        self.lr_2_list = []
        self.lr_10_list = []
        self.lr_20_list = []

    def build(self,net_input,net_label):
        if "output" not in self.net:
            with tf.name_scope("placeholder"):
                self.net["input"] = net_input
                self.net["label"] = net_label # [None, self.h,self.w,1], int32
                self.net["drop_prob"] = tf.Variable(0.5)

            self.net["output"] = self.create_network()
            self.pred()
        return self.net["output"]

    def create_network(self):
        if "init_model_path" in self.config:
            self.load_init_model()
        with tf.name_scope("vgg") as scope:
            # build block

            for scale,key in [ (0.5,"0_dot_5"), (0.75,"0_dot_75"), (1.0, "1_dot_0") ]: # multiscale input
                self.stride["input_%s"%key] = 1
                self.net["input_%s"%key] = tf.image.resize_bilinear(self.net["input"],( int(scale*input_size[0]), int(scale*input_size[1]) ))
                block = self.build_block("input_%s"%key,["conv1_1_%s"%key,"relu1_1_%s"%key,"conv1_2_%s"%key,"relu1_2_%s"%key,"pool1_%s"%key])
                block = self.build_block(block,["conv2_1_%s"%key,"relu2_1_%s"%key,"conv2_2_%s"%key,"relu2_2_%s"%key,"pool2_%s"%key])
                block = self.build_block(block,["conv3_1_%s"%key,"relu3_1_%s"%key,"conv3_2_%s"%key,"relu3_2_%s"%key,"conv3_3_%s"%key,"relu3_3_%s"%key,"pool3_%s"%key])
                block = self.build_block(block,["conv4_1_%s"%key,"relu4_1_%s"%key,"conv4_2_%s"%key,"relu4_2_%s"%key,"conv4_3_%s"%key,"relu4_3_%s"%key,"pool4_%s"%key])
                block = self.build_block(block,["conv5_1_%s"%key,"relu5_1_%s"%key,"conv5_2_%s"%key,"relu5_2_%s"%key,"conv5_3_%s"%key,"relu5_3_%s"%key,"pool5_%s"%key,"pool5a_%s"%key])
                fc = self.build_fc(block,["fc6_%s"%key,"relu6_%s"%key,"drop6_%s"%key,"fc7_%s"%key,"relu7_%s"%key,"drop7_%s"%key,"fc8_%s"%key])

            self.net["output_0_dot_5"] = tf.image.resize_bilinear(self.net["fc8_0_dot_5"],(41,41))
            self.net["output_0_dot_75"] = tf.image.resize_bilinear(self.net["fc8_0_dot_75"],(41,41))
            self.net["output_1_dot_0"] = self.net["fc8_1_dot_0"]

            self.net["output_mixed"] = tf.reduce_max(tf.stack([self.net["output_0_dot_5"],self.net["output_0_dot_75"],self.net["fc8_1_dot_0"]],axis=4),axis=4)

            crf = self.build_crf("output_mixed")

            return self.net["output_mixed"] # NOTE: crf is log-probability

    def build_block(self,last_layer,layer_lists):
        for layer in layer_lists:
            if layer.startswith("conv"):
                if layer[4] != "5":
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        weights,bias = self.get_weights_and_bias(layer)
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                        self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                        last_layer = layer
                if layer[4] == "5":
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        weights,bias = self.get_weights_and_bias(layer)
                        self.net[layer] = tf.nn.atrous_conv2d( self.net[last_layer], weights, rate=2, padding="SAME", name="conv")
                        self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                        last_layer = layer
            if layer.startswith("batch_norm"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer])
                    last_layer = layer
            if layer.startswith("relu"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.nn.relu( self.net[last_layer],name="relu")
                    last_layer = layer
            elif layer.startswith("pool5a"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.nn.avg_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,1,1,1],padding="SAME",name="pool")
                    last_layer = layer
            elif layer.startswith("pool"):
                if layer[4] not in ["4","5"]:
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = 2 * self.stride[last_layer]
                        self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,2,2,1],padding="SAME",name="pool")
                        last_layer = layer
                if layer[4] in ["4","5"]:
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,1,1,1],padding="SAME",name="pool")
                        last_layer = layer
        return last_layer

    def build_fc(self,last_layer, layer_lists):
        for layer in layer_lists:
            if layer.startswith("fc"):
                with tf.name_scope(layer) as scope:
                    weights,bias = self.get_weights_and_bias(layer)
                    if layer.startswith("fc6"):
                        self.net[layer] = tf.nn.atrous_conv2d( self.net[last_layer], weights, rate=12, padding="SAME", name="conv")

                    else:
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                    self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                    last_layer = layer
            if layer.startswith("batch_norm"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer])
                    last_layer = layer
            if layer.startswith("relu"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.nn.relu( self.net[last_layer])
                    last_layer = layer
            if layer.startswith("drop"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.nn.dropout( self.net[last_layer],self.net["drop_prob"])
                    last_layer = layer

        return last_layer

    def build_crf(self,featemap_layer):
        featemap = self.net[featemap_layer]
        h,w = featemap.shape[1],featemap.shape[2]

        origin_image = self.net["input"] + self.data.img_mean
        origin_image_zoomed = tf.image.resize_bilinear(origin_image,(h,w))
        def crf(featemap,image):
            #crf_config = {"g_sxy":3,"g_compat":3,"bi_sxy":80,"bi_srgb":13,"bi_compat":10,"iterations":5} # for test
            crf_config = {"g_sxy":3/12,"g_compat":3,"bi_sxy":80/12,"bi_srgb":13,"bi_compat":10,"iterations":5} # for train, in previous iter, probability is not accurate, so we should weaken the weight of position

            batch_size = featemap.shape[0]
            image = image.astype(np.uint8)
            ret = np.zeros(featemap.shape,dtype=np.float32)
            for i in range(batch_size):
                ret[i,:,:,:] = crf_inference(image[i],crf_config,self.category_num,featemap[i],use_log=True)

            ret[ret < self.min_prob] = self.min_prob
            ret /= np.sum(ret,axis=3,keepdims=True)
            ret = np.log(ret)
            return ret.astype(np.float32)
    
        layer = "crf"
        crf = tf.py_func(crf,[featemap,origin_image_zoomed],tf.float32) # shape [N, h, w, C], RGB or BGR doesn't matter
        self.net[layer] = crf

        return layer

    def get_weights_and_bias(self,layer,shape=None):
        print("layer: %s" % layer)
        for key in ["0_dot_5", "0_dot_75", "1_dot_0"]:
            if layer.endswith(key):
                origin_layer = layer
                layer = layer[:-len(key)-1]
        if layer in self.weights:
            return self.weights[layer]
        if shape is not None:
            pass
        elif layer.startswith("conv"):
            shape = [3,3,0,0]
            if layer == "conv1_1":
                shape[2] = 3
            else:
                shape[2] = 64 * self.stride[origin_layer]
                if shape[2] > 512: shape[2] = 512
                if layer in ["conv2_1","conv3_1","conv4_1"]: shape[2] = int(shape[2]/2)
            shape[3] = 64 * self.stride[origin_layer]
            if shape[3] > 512: shape[3] = 512
        elif layer.startswith("fc"):
            if layer == "fc6":
                shape = [3,3,512,1024]
            if layer == "fc7":
                shape = [1,1,1024,1024]
            if layer == "fc8": 
                shape = [1,1,1024,self.category_num]
        if "init_model_path" not in self.config:
            init = tf.random_normal_initializer(stddev=0.01)
            weights = tf.get_variable(name="%s_weights" % layer,initializer=init, shape = shape)
            init = tf.constant_initializer(0)
            bias = tf.get_variable(name="%s_bias" % layer,initializer=init, shape = [shape[-1]])
        else:
            if layer in ["fc8"]:
                init = tf.contrib.layers.xavier_initializer(uniform=True)
            else:
                init = tf.constant_initializer(self.init_model[layer]["w"])
            weights = tf.get_variable(name="%s_weights" % layer,initializer=init,shape = shape)
            if layer in ["fc8"]:
                init = tf.constant_initializer(0)
            else:
                init = tf.constant_initializer(self.init_model[layer]["b"])
            bias = tf.get_variable(name="%s_bias" % layer,initializer=init,shape = [shape[-1]])
        self.weights[layer] = (weights,bias)
        if layer in ["fc8"]:
            self.lr_10_list.append(weights)
            self.lr_20_list.append(bias)
        else:
            self.lr_1_list.append(weights)
            self.lr_2_list.append(bias)
        self.trainable_list.append(weights)
        self.trainable_list.append(bias)
        self.variables["total"].append(weights)
        self.variables["total"].append(bias)

        return weights,bias

    def getloss(self):
        label_scale = tf.image.resize_nearest_neighbor( tf.cast(self.net["label"],tf.int32), (41,41) )

        total_loss = 0
        for key in ["0_dot_5", "0_dot_75",  "1_dot_0", "mixed"]:
            label,output = self.remove_ignore_label(label_scale,self.net["output_%s"%key])
            total_loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=output))
        return total_loss


    def train(self,base_lr,weight_decay,momentum,batch_size,epoches):
        gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.80))
        self.sess = tf.Session(config=gpu_options)
        #self.sess = tf.Session()
        data_x,data_y,id_of_image,iterator_train = self.data.next_batch(category="train",batch_size=batch_size,epoches=-1)
        self.build(net_input=data_x,net_label=data_y)
        self.pre_train(base_lr,weight_decay,momentum,batch_size,save_layers=["input","output","label","pred","drop_prob"])
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(iterator_train.initializer)

            if self.config.get("model_path",False) is not False:
                print("start to load model: %s" % self.config.get("model_path"))
                print("before l2 loss:%f" % self.sess.run(self.l2loss["total"]))
                self.restore_from_model(self.saver["norm"],self.config.get("model_path"),checkpoint=False)
                print("model loaded ...")
                print("after l2 loss:%f" % self.sess.run(self.l2loss["total"]))

            start_time = time.time()
            print("start_time: %f" % start_time)
            print("config -- lr:%f weight_decay:%f momentum:%f batch_size:%f epoches:%f" % (base_lr,weight_decay,momentum,batch_size,epoches))

            epoch,i = 0.0,0
            iterations_per_epoch_train = self.data.get_data_len() // batch_size
            while epoch < epoches:
                if i == 0: # to protect restore
                    self.sess.run(tf.assign(self.net["lr"],base_lr))
                    self.sess.run(self.net["accum_gradient_clean"])
                if i == 10*iterations_per_epoch_train:
                    new_lr = 3e-4
                    print("save model before new_lr:%f" % new_lr)
                    self.saver["lr"].save(self.sess,self.config.get("saver_path","saver"),global_step=i)
                    self.sess.run(tf.assign(self.net["lr"],new_lr))
                    base_lr = new_lr
                if i == 15*iterations_per_epoch_train:
                    new_lr = 1e-4
                    print("save model before new_lr:%f" % new_lr)
                    self.saver["lr"].save(self.sess,self.config.get("saver_path","saver"),global_step=i)
                    self.sess.run(tf.assign(self.net["lr"],new_lr))
                    base_lr = new_lr
                if i == 20*iterations_per_epoch_train:
                    new_lr = 3e-5
                    print("save model before new_lr:%f" % new_lr)
                    self.saver["lr"].save(self.sess,self.config.get("saver_path","saver"),global_step=i)
                    self.sess.run(tf.assign(self.net["lr"],new_lr))
                    base_lr = new_lr


                self.sess.run(self.net["accum_gradient_accum"])
                #self.sess.run(self.net["train_op"])
                if i % self.accum_num == self.accum_num - 1:
                    _ = self.sess.run(self.net["accum_gradient_update"])
                    _ = self.sess.run(self.net["accum_gradient_clean"])

                if i%1000 == 0:
                    l2loss,normloss,loss,lr = self.sess.run([self.l2loss["total"],self.loss["norm"],self.loss["total"],self.net["lr"]])
                    print("epoch:%f, iteration:%f, lr:%f, loss:%f l2:%f, norm:%f" % (epoch,i,lr,loss,l2loss,normloss))

                if i%3000 == 2999:
                    self.saver["norm"].save(self.sess,self.config.get("saver_path","saver"),global_step=i)
                i+=1
                epoch = i / iterations_per_epoch_train

                #if i >= 50:
                #    epoch += 300
                epoch += 300

            end_time = time.time()
            print("end_time:%f" % end_time)
            print("duration time:%f" %  (end_time-start_time))
            self.saver["norm"].save(self.sess,self.config.get("saver_path","saver"),global_step=0)

if __name__ == "__main__":
    batch_size = 2 # the actual batch size is  batch_size * accum_num
    input_size = (321,321)
    category_num = 21
    epoches = 25
    data = dataset({"batch_size":batch_size,"input_size":input_size,"epoches":epoches,"category_num":category_num,"categorys":["train"]})
    deeplab = DeepLab_largefov({"data":data,"batch_size":batch_size,"input_size":input_size,"epoches":epoches,"category_num":category_num,"init_model_path":"./model/largefov.npy","accum_num":64,"saver_path":"saver/norm"})

    lr = 1e-3
    deeplab.train(base_lr=lr,weight_decay=5e-4,momentum=0.9,batch_size=batch_size,epoches=epoches)
