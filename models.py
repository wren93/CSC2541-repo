
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
import json
from utils import build_pretrain_embedding, load_embeddings
from math import floor


class WordRep(nn.Module):
    def __init__(self, args, Y, dicts):
        super(WordRep, self).__init__()

        if args.embed_file:
            print("loading pretrained embeddings from {}".format(args.embed_file))
            if args.use_ext_emb:
                pretrain_word_embedding, pretrain_emb_dim = build_pretrain_embedding(args.embed_file, dicts['w2ind'],
                                                                                     True)
                W = torch.from_numpy(pretrain_word_embedding)
            else:
                W = torch.Tensor(load_embeddings(args.embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            self.embed = nn.Embedding(len(dicts['w2ind']) + 2, args.embed_size, padding_idx=0)
        self.feature_size = self.embed.embedding_dim

        self.embed_drop = nn.Dropout(p=args.dropout)

        self.conv_dict = {1: [self.feature_size, args.num_filter_maps],
                     2: [self.feature_size, 100, args.num_filter_maps],
                     3: [self.feature_size, 150, 100, args.num_filter_maps],
                     4: [self.feature_size, 200, 150, 100, args.num_filter_maps]
                     }


    def forward(self, x, target, text_inputs):
        features = [self.embed(x)]

        x = torch.cat(features, dim=2)

        x = self.embed_drop(x)
        return x


class Decoder(nn.Module):
    """
    Decoder part: per-label attention layer and classifier
    """
    def __init__(self, args, Y, dicts, input_size):
        super(Decoder, self).__init__()

        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)


        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()


    def forward(self, x, target, text_inputs):
        # attention?
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)

        m = alpha.matmul(x)

        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        loss = self.loss_function(y, target)
        return y, loss, alpha, m


class CNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(CNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        filter_size = int(args.filter_size)


        self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
        xavier_uniform(self.conv.weight)

        self.output_layer = Decoder(args, Y, dicts, args.num_filter_maps)


    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        x = torch.tanh(self.conv(x).transpose(1, 2))

        y, loss, alpha, m = self.output_layer(x, target, text_inputs)
        return y, loss, alpha, m

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiCNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(MultiCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        if args.filter_size.find(',') == -1:
            self.filter_num = 1
            filter_size = int(args.filter_size)
            self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
            xavier_uniform(self.conv.weight)
        else:
            filter_sizes = args.filter_size.split(',')
            self.filter_num = len(filter_sizes)
            self.conv = nn.ModuleList()
            for filter_size in filter_sizes:
                filter_size = int(filter_size)
                tmp = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                      padding=int(floor(filter_size / 2)))
                xavier_uniform(tmp.weight)
                self.conv.add_module('conv-{}'.format(filter_size), tmp)

        self.output_layer = Decoder(args, Y, dicts, self.filter_num * args.num_filter_maps)



    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        if self.filter_num == 1:
            x = torch.tanh(self.conv(x).transpose(1, 2))
        else:
            conv_result = []
            for tmp in self.conv:
                conv_result.append(torch.tanh(tmp(x).transpose(1, 2)))
            x = torch.cat(conv_result, dim=2)

        y, loss, alpha, m = self.output_layer(x, target, text_inputs)

        return y, loss, alpha, m

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

class ResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(ResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        conv_dimension = self.word_rep.conv_dict[args.conv_layer]
        for idx in range(args.conv_layer):
            tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], int(args.filter_size), 1, True, args.dropout)
            self.conv.add_module('conv-{}'.format(idx), tmp)

        self.output_layer = Decoder(args, Y, dicts, args.num_filter_maps)


    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        for conv in self.conv:
            x = conv(x)
        x = x.transpose(1, 2)

        y, loss, alpha, m = self.output_layer(x, target, text_inputs)

        return y, loss, alpha, m

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = Decoder(args, Y, dicts, self.filter_num * args.num_filter_maps)


    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)

        y, loss, alpha, m = self.output_layer(x, target, text_inputs)

        return y, loss, alpha, m

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

import os
from transformers import BertConfig, BertModel
class BertStandard(nn.Module):
    def __init__(self, args, Y):
        super(BertStandard, self).__init__()

        print("loading pretrained bert from {}".format(args.bert_dir))
        config_file = os.path.join(args.bert_dir, 'config.json')
        self.config = BertConfig.from_json_file(config_file)
        print("Model config {}".format(self.config))
        self.bert = BertModel.from_pretrained(args.bert_dir)
        
        # decoder
        self.decoder = Decoder(args, Y, None, self.config.hidden_size)

    def forward(self, input_ids, token_type_ids, attention_mask, target):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        encoder_output = outputs.last_hidden_state

        y, loss, alpha, m = self.decoder(encoder_output, target, None)
        return y, loss, alpha, m

    def init_bert_weights(self, module):

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def freeze_net(self):
        pass


from transformers import XLNetModel, XLNetConfig
from transformers.modeling_utils import SequenceSummary
class XLNet(nn.Module):

    def __init__(self, args, Y):
        super(XLNet, self).__init__()

        print("loading pretrained xlnet from {}".format(args.xlnet_dir))
        config_file = os.path.join(args.xlnet_dir, 'config.json')
        self.config = XLNetConfig.from_json_file(config_file)
        print("Model config {}".format(self.config))
        self.xlnet = XLNetModel.from_pretrained(args.xlnet_dir)

        # decoder
        self.decoder = Decoder(args, Y, None, self.config.d_model)

        self.sequence_summary = SequenceSummary(self.config)

        self.classifier = nn.Linear(self.config.d_model, Y)

    def forward(self, input_ids, token_type_ids, attention_mask, target):
        xlnet_output = self.xlnet(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False)
        output = xlnet_output[0]

        # output = self.sequence_summary(output)
        # y = self.classifier(output)
        # loss = F.binary_cross_entropy_with_logits(y, target)

        y, loss, alpha, m = self.decoder(output, target, None)
        return y, loss, alpha, m

    def freeze_net(self):
        pass


from transformers import LongformerModel, LongformerConfig
class LongformerClassifier(nn.Module):

    def __init__(self, args, Y):
        super(LongformerClassifier, self).__init__()

        print("loading pretrained longformer from {}".format(args.longformer_dir))
        config_file = os.path.join(args.longformer_dir, 'config.json')
        self.config = LongformerConfig.from_json_file(config_file)
        print("Model config {}".format(self.config))
        self.longformer = LongformerModel.from_pretrained(args.longformer_dir, gradient_checkpointing=True)

        # decoder
        self.decoder = Decoder(args, Y, None, self.config.hidden_size)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, Y)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, target, global_attention_mask=None):
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            # global_attention_mask[:, 0] = 1 # this line should be commented if using decoder

        longformer_output = self.longformer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=False
        )

        # pooled_output = longformer_output[1]
        # output = self.dropout(output)
        # y = self.classifier(output)
        # loss = F.binary_cross_entropy_with_logits(y, target)

        output = longformer_output[0]
        y, loss, alpha, m = self.decoder(output, target, None)
        return y, loss, alpha, m

    def freeze_net(self):
        pass


def pick_model(args, dicts):
    Y = len(dicts['ind2c']) # total number of ICD codes
    if args.model == 'CNN':
        model = CNN(args, Y, dicts)
    elif args.model == 'MultiCNN':
        model = MultiCNN(args, Y, dicts)
    elif args.model == 'ResCNN':
        model = ResCNN(args, Y, dicts)
    elif args.model == 'MultiResCNN':
        model = MultiResCNN(args, Y, dicts)
    elif args.model == 'bert':
        model = BertStandard(args, Y)
    elif args.model == 'xlnet':
        model = XLNet(args, Y)
    elif args.model == 'longformer':
        model = LongformerClassifier(args, Y)
    else:
        raise RuntimeError("wrong model name")

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.tune_wordemb == False:
        model.freeze_net()
    if len(args.gpu_list) == 1 and args.gpu_list[0] != -1: # single card training
        model.cuda()
    elif len(args.gpu_list) > 1: # multi-card training
        model = nn.DataParallel(model, device_ids=args.gpu_list)
        model = model.to(f'cuda:{model.device_ids[0]}')
    return model
