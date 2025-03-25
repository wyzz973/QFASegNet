class Config_picai:
    def __init__(self, **kwargs):

        self.n_epochs = 100
        self.epoch_start = 0

        self.img_height = 192
        self.img_width = 192
        self.batch_size = 1
        self.total_slices = 33
        self.slice_nums = 3

        self.class_nums = 3
        self.best_dsc = 0

        """codebook"""
        self.n_e = 256
        self.e_dim = 16      #16 /64 /256
        self.quant_nums = 16  #2 /4 /8 /16 /32
        self.beta = 0.25


        """network"""
        self.in_channel = 64
        self.nums_res_block = 2
        self.ch_mult = [1, 2, 4, 8]
        self.norm_type = 'group'
        self.act_type = 'swish'
        self.independent_layer_count = 1

        """align_loss"""
        self.displacement = True
        self.displace_scale = [1]  #[0] /[1] /[1,2] /[1,2,3] /[1,2,4] /[1,3] /[1,3,5]
        self.temperature = 1
        self.align_type = 'js'



        """parameter"""
        self.learning_rate = 0.001
        self.lambda_target = 0.1   #0.1 /0.5 /1
        self.lambda_vq_feature = 0.01 # 0.01 /0.1 /1
        self.lambda_gan = 0.1  #0.1 /0.01
        self.lambda_align_loss = 1 #0.1 /0.01 /1
        self.lambda_vq = 1
        self.lambda_D = 1
        self.lambda_re = 1




        """dataset"""
        self.source_modality = 't2w'
        self.target_modality = 'adc'
        self.dataset = 'picai'
        self.dataroot = '../../dataset/picai'

        self.device = 'cuda'
        self.save_images = True
        self.version = f'sd'

        self.seed = 973
        self.early_stop = False
        for key, value in kwargs.items():
            setattr(self, key, value)

    def print_attributes(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")

    def print_opt(self):
        info = (f'--------------{self.version}:parameter weighting-----------------\n'
                f'n_e:{self.n_e}\te_dim:{self.e_dim}\tbeta:{self.beta}\tin_channel:{self.in_channel}\n'
                f'lambda_target:{self.lambda_target}\tquant_nums:{self.quant_nums}\tlambda_vq_feature:{self.lambda_vq_feature}\n'
                f'lambda_gan:{self.lambda_gan}\t\tlambda_align_loss:{self.lambda_align_loss}\tlambda_vq:{self.lambda_vq}\n'
                f'displace_scale:{self.displace_scale}\t\tch_mult:{self.ch_mult}\t\talign_type:{self.align_type}\n'
                )

        return info

    def filter_config_for_network(self,decoder=False):
        if not decoder:
            param_names = [
                'slice_nums', 'in_channel', 'ch_mult', 'e_dim',
                'nums_res_block', 'norm_type', 'act_type',
                'quant_nums'
            ]
        else:
            param_names = [
                'slice_nums', 'in_channel', 'ch_mult', 'e_dim',
                'nums_res_block', 'norm_type', 'act_type','class_nums',
                'quant_nums'
            ]
        params = {param: getattr(self, param) for param in param_names if hasattr(self, param)}
        return params



    def to_dict(self):
        # 使用 `vars(self)` 或 `self.__dict__` 来获取实例属性和值的字典
        return vars(self)


