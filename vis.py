from functools import partial
from options import args
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from utils import load_lookups, prepare_instance, prepare_instance_bert, prepare_instance_xlnet, \
    MyDataset, my_collate, my_collate_bert, prepare_instance_longformer
from models import pick_model

if __name__ == "__main__":
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
    selected_label_list = [2534, 8806]

    if args.model.find("bert") != -1 or args.model.find("xlnet") != -1 or args.model.find("longformer") != -1:
        collate_func = my_collate_bert
    else:
        collate_func = partial(my_collate, use_elmo=args.use_elmo)
    test_loader = DataLoader(MyDataset(test_instances), 1, shuffle=False, collate_fn=collate_func, num_workers=args.num_workers, pin_memory=True)

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
    for i in range(len(feature_list)):
        X = np.vstack(feature_list[i])
        X = np.asarray(X, dtype='float64')
        y = label_list[i]
        X_embedded = TSNE(n_components=2, init='pca', n_iter=5000).fit_transform(X)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
        plt.show()
