from .model import TSNet
from .dataset import UCRDataLoaderBuilder

import torch
from torch import optim, nn
from torchland.trainer.trainer import NetworkTrainer, TrainStage


class TSNetTrainer(NetworkTrainer):
    def __init__(self, config: dict):
        super().__init__(epoch=config.get('epoch', 30))
        self.lr_init = config.get('lr_init', 3e-4)
        self.batch_size = config.get('batch_size', 5)
        data_root = config.get('data_root', 'data_in')
        data_name = config.get('data_name', None)  # must be specified
        self.sample_length = config.get('sample_length', 100)

        self.input_size = (1, self.sample_length)

        model = TSNet(in_channel=1, middle_channel=50, out_channel=10, num_layers=5)
        self.add_model('tsnet', model, input_size=self.input_size)
        self.add_optimizer(
            'adam', optim.Adam(model.parameters(), lr=self.lr_init, weight_decay=0.01))
        self.add_criterion('cosine_embedding_loss', nn.CosineEmbeddingLoss(margin=0.3))

        self.dataloader_maker = UCRDataLoaderBuilder(
            data_root, data_name, self.batch_size, sample_length=self.sample_length)
        self.set_dataloaders(self.dataloader_maker)

    def num_samples(self, train_stage: TrainStage):
        if train_stage == TrainStage.TRAIN:
            return self.dataloader_maker.num_train
        elif train_stage == TrainStage.VALIDATE:
            return self.dataloader_maker.num_val
        elif train_stage == TrainStage.TEST:
            return self.dataloader_maker.num_test
        raise ValueError(f'Invalid train_stage value: {train_stage}')

    def run_step(self,
                 models,
                 criteria,
                 optimizers,
                 input_,
                 train_stage: TrainStage,
                 *args, **kwargs):
        class_labels, anchors, positive_samples, negative_samples = input_

        # get loss functions
        cosine_embedding_loss = criteria.cosine_embedding_loss

        encoder = models.tsnet.model

        z_anchors: torch.FloatTensor = encoder(anchors)
        z_positive = encoder(positive_samples)
        z_negative = encoder(negative_samples)

        # calculate loss values
        target_ones = torch.ones(z_anchors.size(0))
        positive_loss = cosine_embedding_loss(z_anchors, z_positive, target=target_ones)
        negative_loss = cosine_embedding_loss(z_anchors, z_negative, target=-target_ones)
        total_loss = positive_loss + negative_loss

        if train_stage == TrainStage.TRAIN:
            adam = optimizers.adam
            adam.zero_grad()
            total_loss.backward()
            adam.step()

        outputs = {
            'class_labels': class_labels,
            'z_anchors': z_anchors,
            'z_positive': z_positive,
            'z_negative': z_negative,
        }

        loss_val = {
            'total_loss': total_loss,
            'positive_loss': positive_loss,
            'negative_loss': negative_loss,
        }

        return outputs, loss_val

    def pre_epoch_finish(
            self,
            input_: torch.Tensor,
            output: tuple,
            metric_manager,
            train_stage: TrainStage):
        # draw embeddings of test
        if train_stage == TrainStage.TEST:
            class_label_list = []
            embeddings = []
            samples_list = []
            model = self._models.tsnet.model

            for step, input_ in enumerate(self._dataloaders.test_loader):
                class_labels, samples, _, _ = input_
                embd = model(samples).squeeze()

                embeddings.append(embd)
                class_label_list.append(class_labels)
                samples_list.append(samples[:, :, :100])

            # concatenate all by batches
            embeddings_cat = torch.cat(embeddings, dim=0).squeeze()
            class_labels_cat = torch.cat(class_label_list, dim=0)
            samples_cat = torch.cat(samples_list, dim=0).squeeze()

            scale_factor = 5
            samp_vals = torch.floor((samples_cat * scale_factor + 20)).type(torch.long).clamp(min=0, max=99)
            print(samp_vals.min())
            print(samp_vals.max())
            img_tensors = []
            for samp_val in samp_vals:
                num_samps = samp_val.size(0)
                one_hot = torch.zeros((num_samps, 100))
                img_vector = one_hot.scatter_(1, samp_val.unsqueeze(dim=1), 1)
                img_vector = img_vector.unsqueeze(dim=0)
                img_tensors.append(torch.cat((img_vector, img_vector, img_vector), dim=0).unsqueeze(dim=0))

            print(class_labels_cat[:20])
            print(embeddings_cat[:20])

            self._writer.add_embedding(
                embeddings_cat,
                metadata=class_labels_cat,
                label_img=torch.cat(img_tensors, dim=0),
                tag=f'embeddings/{self._epoch}',
                global_step=self._epoch,
            )

    @staticmethod
    def make_performance_metric(input_: torch.Tensor, output, loss) -> dict:
        return {
            'total_loss': loss['total_loss'].item(),
            'positive_loss': loss['positive_loss'].item(),
            'negative_loss': loss['negative_loss'].item(),
        }


if __name__ == '__main__':
    test_config = {
        'epoch': 150,
        'data_name': 'ChlorineConcentration',
        'data_root': '/Users/dansuh/datasets/ucr-archive/UCRArchive_2018/',
        'sample_length': 150,
    }
    trainer = TSNetTrainer(config=test_config)
    trainer.fit()
