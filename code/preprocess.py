import io
import numpy as np

def load_data(fname, class_map=None):
    def label_to_class(labels):
        class_map = {}
        class_id = 0
        for i, label in enumerate(labels):
            if label not in class_map:
                class_map[label] = class_id
                class_id +=1
        return class_map
    with open(fname, encoding="utf-8") as f:
        lines = f.readlines()
    sents = []
    labels = []
    for line in lines:
        label, sent = line.split('|||')
        sents.append(sent)
        labels.append(label)

    if class_map is None:
        class_map = label_to_class(labels)
    inv_class_map = {}
    for k, v in class_map.items():
        inv_class_map[v] = k

    for i in range(len(labels)):
        if labels[i] == 'UNK ':
            labels[i] = None
            continue
        if labels[i] == 'Media and darama ':
            labels[i] = 'Media and drama '
        labels[i] = class_map[labels[i]]
    
    return sents, labels, inv_class_map, class_map

def load_vectors():
    fname = '../data/wiki-news-300d-1M.vec'
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(token) for token in tokens[1:]]
    return data

def process_sentences(sents, embs):
    for i, sent in enumerate(sents):
        sent_embs = []
        for w in sent.split():
            if w in embs:
                sent_embs.append(np.expand_dims(np.array(embs[w]), 0))
        sent_embs = np.concatenate(sent_embs, axis=0)
        sents[i] = sent_embs


if __name__=='__main__':
    
    train_sents, train_labels, id_to_class, class_to_id = load_data('../data/topicclass_train.txt')
    val_sents, val_labels, _, _ = load_data('../data/topicclass_valid.txt', class_to_id)
    test_sents, _, _, _ = load_data('../data/topicclass_test.txt', class_to_id)
    
    embs = load_vectors()
    process_sentences(train_sents, embs)
    process_sentences(val_sents, embs)
    process_sentences(test_sents, embs)
    
    np.save('trainx.npy', train_sents)
    np.save('trainy.npy', train_labels)
    np.save('valx.npy', val_sents)
    np.save('valy.npy', val_labels)
    np.save('testx.npy', test_sents)
    with open('ids.json', 'w') as f:
        json.dump(id_to_class, f)