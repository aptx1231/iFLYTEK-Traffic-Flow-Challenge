from libcity.model.abstract_model import AbstractModel
from libcity.model import loss
from functools import partial


class AbstractTrafficStateModel(AbstractModel):

    def __init__(self, config, data_feature):
        self.data_feature = data_feature
        self.loss = config.get('loss', 'mse')
        if self.loss == 'mse':
            self.lf = loss.masked_mse_torch
        elif self.loss == 'mae':
            self.lf = loss.masked_mae_torch
        elif self.loss == 'masked_mae':
            self.lf = partial(loss.masked_mae_torch, null_val=0)
        elif self.loss == 'huber':
            delta = config.get('delta', 1.0)
            self.lf = partial(loss.huber_loss, delta=delta)
        elif self.loss == 'mse_diff':
            self.lf = loss.mse_diff
        super().__init__(config, data_feature)

    def predict(self, batch):
        """
        输入一个batch的数据，返回对应的预测值，一般应该是**多步预测**的结果，一般会调用nn.Moudle的forward()方法

        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: predict result of this batch
        """

    def calculate_loss(self, batch):
        """
        输入一个batch的数据，返回训练过程的loss，也就是需要定义一个loss函数

        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: return training loss
        """
