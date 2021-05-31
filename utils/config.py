from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.

class Config:
    #num of Classes
    num_class = 11 #2

    #score_thresh
    visualize_score_thresh = 0.7

    # data
    SDL_data_dir = 'dataset/SDL'
    min_size = 600  # image resize
    max_size = 600  # image resize
    num_workers = 0
    test_num_workers = 0

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3


    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 10  # vis every N iter

    # preset
    data = 'SDL'
    pretrained_model = 'vgg16'

    # training
    epoch = 1


    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    # test_num = 10000
    test_num = 51
    # model
    # load_path = r'E:\10277\OneDrive\Python\Homework\20210527\Intelligent diagnosis of spine diseases\net_pro.pth'
    load_path = 'net/net_pro.pth'
    # load_path = None
    # save_path = r'E:\10277\OneDrive\Python\Homework\20210527\Intelligent diagnosis of spine diseases\net_pro.pth'
    save_path = 'net/net_pro.pth'

    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    # caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
