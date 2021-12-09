import sys
from functools import partial
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from utils import load_lookups, prepare_instance, prepare_instance_bert, prepare_instance_xlnet, \
    MyDataset, my_collate, my_collate_bert, prepare_instance_longformer
from models import pick_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-MODEL_DIR', type=str, default='./models')
    parser.add_argument('-DATA_DIR', type=str, default='./data')
    parser.add_argument('-MIMIC_3_DIR', type=str, default='./data/mimic3')
    parser.add_argument('-MIMIC_2_DIR', type=str, default='./data/mimic2')

    parser.add_argument("-data_path", type=str, default='./data/mimic3/train_full.csv')
    parser.add_argument("-vocab", type=str, default='./data/mimic3/vocab.csv')
    parser.add_argument("-Y", type=str, default='full', choices=['full', '50'])
    parser.add_argument("-version", type=str, choices=['mimic2', 'mimic3'], default='mimic3')
    parser.add_argument("-MAX_LENGTH", type=int, default=3200)

    # model
    parser.add_argument("-model", type=str, choices=['CNN', 'MultiCNN', 'ResCNN', 'MultiResCNN', 'bert', 'xlnet', 'longformer'], default='longformer')
    parser.add_argument("-model_name", type=str, default="Longformer-3200")
    parser.add_argument("-filter_size", type=str, default="3,5,9,15,19,25")
    parser.add_argument("-num_filter_maps", type=int, default=50)
    parser.add_argument("-conv_layer", type=int, default=1)
    parser.add_argument("-embed_file", type=str, default='./data/mimic3/processed_full.embed')
    parser.add_argument("-test_model", type=str, default="./models/Longformer_3200_full.pth")
    parser.add_argument("-use_ext_emb", action="store_const", const=True, default=False)
    parser.add_argument("-gpu", type=str, default='0', help='-1 if not using gpu, use comma to separate multiple gpus')
    parser.add_argument("-num_workers", type=int, default=16)
    parser.add_argument("-tune_wordemb", action="store_const", const=True, default=False)
    parser.add_argument('-random_seed', type=int, default=1, help='0 if randomly initialize the model, other if fix the seed')
    parser.add_argument("-use_elmo", action="store_const", const=True, default=False)
    parser.add_argument("-dropout", type=float, default=0.2)

    # bert
    parser.add_argument("-bert_dir", type=str, default='bert/bert-base-uncased')

    # xlnet
    parser.add_argument("-xlnet_dir", type=str, default='bert/clinical_xlnet_pytorch')

    # longformer
    parser.add_argument("-longformer_dir", type=str, default='bert/longformer-base-4096')

    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command

    # gpu settings
    args.gpu_list = [int(idx) for idx in args.gpu.split(',')]
    args.gpu_list = [i for i in range(len(args.gpu_list))] if args.gpu_list[0] >= 0 else [-1]

    # load vocab and other lookups
    print("loading lookups...")
    dicts = load_lookups(args)

    model = pick_model(args, dicts)

    if args.model.find("bert") != -1:
        prepare_instance_func = prepare_instance_bert
    elif args.model.find("xlnet") != -1:
        prepare_instance_func = prepare_instance_xlnet
    elif args.model.find("longformer") != -1:
        prepare_instance_func = prepare_instance_longformer
    else:
        prepare_instance_func = prepare_instance
    
    test_instances = prepare_instance_func(dicts, args.data_path.replace('train','test'), args, args.MAX_LENGTH)
    print("test_instances {}".format(len(test_instances)))

    label_count = np.zeros(len(dicts['c2ind']))
    for i in range(len(test_instances)):
        label_count += test_instances[i]['label']
    
    sorted_index = np.argsort(-label_count)

    selected_label_list = list(sorted_index[:1])
    selected_label_list = [2534] # 1134

    if args.model.find("bert") != -1 or args.model.find("xlnet") != -1 or args.model.find("longformer") != -1:
        collate_func = my_collate_bert
    else:
        collate_func = partial(my_collate, use_elmo=args.use_elmo)
    test_loader = DataLoader(MyDataset(test_instances), 1, shuffle=False, collate_fn=collate_func, num_workers=0, pin_memory=True)

    gpu = args.gpu_list
    
    feature_list = [[] for i in range(len(selected_label_list))]
    label_list = [[] for i in range(len(selected_label_list))]

    model.eval()

    # loader
    data_iter = iter(test_loader)
    num_iter = len(test_loader)
    for i in range(num_iter):
        with torch.no_grad():

            if args.model.find("bert") != -1 or args.model.find("xlnet") != -1 or args.model.find("longformer") != -1:
                inputs_id, segments, masks, labels = next(data_iter)

                inputs_id, segments, masks, labels = torch.LongTensor(inputs_id), torch.LongTensor(segments), \
                                                     torch.LongTensor(masks), torch.FloatTensor(labels)

                if gpu[0] >= 0:
                    inputs_id, segments, masks, labels = inputs_id.cuda(), segments.cuda(), masks.cuda(), labels.cuda()

                output, loss, alpha, m = model(inputs_id, segments, masks, labels)
            else:

                inputs_id, labels, text_inputs = next(data_iter)

                inputs_id, labels, = torch.LongTensor(inputs_id), torch.FloatTensor(labels)

                if gpu[0] >= 0:
                    inputs_id, labels, text_inputs = inputs_id.cuda(), labels.cuda(), text_inputs.cuda()

                output, loss, alpha, m = model(inputs_id, labels, text_inputs)

            for j in range(len(selected_label_list)):
                feature_list[j].append(m[0, selected_label_list[j], :].cpu().numpy())
                label_list[j].append(labels[0][selected_label_list[j]].cpu().item())
    
    # T-SNE and visualization
    fig = plt.figure(figsize=(8, 4))
    for i in range(len(feature_list)):
        X = np.vstack(feature_list[i])
        X = np.asarray(X, dtype='float64')
        y = np.array(label_list[i])
        X_embedded = TSNE(n_components=2, init='pca', n_iter=5000).fit_transform(X)

        y_0 = np.where(y == 0)[0]
        X_embedded_0 = X_embedded[y_0, :]

        y_1 = np.where(y == 1)[0]
        X_embedded_1 = X_embedded[y_1, :]

        print("save t-sne data for " + args.model_name + " to ./plot/ ...")

        np.savetxt("./plot/" + args.model_name + "_X_0.txt", X_embedded_0)
        np.savetxt("./plot/" + args.model_name + "_X_1.txt", X_embedded_1)
        np.savetxt("./plot/" + args.model_name + "_y_0.txt", y_0)
        np.savetxt("./plot/" + args.model_name + "_y_1.txt", y_1)
        
    #     ax = fig.add_subplot(1, len(feature_list), i + 1)
    #     ax.scatter(X_embedded_0[:, 0], X_embedded_0[:, 1], color="blue", alpha=0.5)
    #     ax.scatter(X_embedded_1[:, 0], X_embedded_1[:, 1], color="red", alpha=0.5)
    #     ax.legend(["Code is assigned to ", "0"])
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_title(args.model_name + " - " + dicts['ind2c'][selected_label_list[i]])

    # plt.show()
