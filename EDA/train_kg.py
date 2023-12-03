from torchkge.models.bilinear import *
from torchkge.models.translation import *
from torchkge.models.deep import ConvKBModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge.data_structures import KnowledgeGraph
from tqdm.autonotebook import tqdm
import pandas as pd
import argparse
from torch import cuda
from torch.optim import Adam
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import pdb
import ast
import numpy as np
import json



def load_models(args,n_entities, n_relations):
    if args.model == 'ComplEx':
        model = ComplExModel(emb_dim=args.embed_dim, n_entities=n_entities, n_relations=n_relations)
    elif args.model == 'analogy':
        model = AnalogyModel(emb_dim=args.embed_dim, n_entities=n_entities, n_relations=n_relations, scalar_share=args.scalar_share)
    elif args.model == 'TransE':
        model = TransEModel(emb_dim=args.embed_dim, n_entities=n_entities, dissimilarity_type='L2')
    elif args.model == 'TransH':
        model = TransHModel(emb_dim=args.embed_dim, n_entities=n_entities)
    elif args.model == 'TransR':
        model = TransRModel(ent_emb_dim = args.ent_emb_dim, rel_emb_dim=args.rel_embed_dim, n_entities=n_entities)
    elif args.model == 'convKB':
        model = ConvKBModel(emb_dim=args.embed_dim, n_entities=n_entities, n_filters = args.n_filters)
    return model

def str2dtype(x, dtype):
    try: return ast.literal_eval(x)
    except:
        if dtype =='list' :return []
        elif dtype =='dict' :return {}
        else: return ''


def to_kg():

    df = pd.read_csv('/home/dhkim/server_front/winery_AI/winery/data/item_data_expand.csv', encoding = 'utf-8-sig')
    wine = pd.read_csv('/home/dhkim/server_front/winery_AI/winery/data/wine_df.csv', encoding = 'utf-8-sig')
    
    knowledge_df = pd.DataFrame(columns=['from','rel','to'])

    notes = ["Red_Fruit","Tropical","Tree_Fruit","Oaky",
                "Ageing","Black_Fruit","Citrus","Dried_Fruit","Earthy",
                "Floral","Microbio","Spices","Vegetal"] 
    for col in ['grape','pairing']:
        df[col] = df[col].apply(lambda x: str2dtype(x, 'list'))

    for col in wine.columns:
        wine.rename(columns={col:col.replace(' ','_')}, inplace = True)
    for col in notes:
        wine[col+'_child'] = wine[col+'_child'].apply(lambda x: str2dtype(x, 'dict'))


    relations = [('item_id','region','region'), ('item_id','made_by','winery'),('item_id','winetype','winetype'),
                ('region','located','country'),('winery','located','region')]
    
    kg = []
    for i in tqdm(range(len(df))):
        for from_, rel, to in relations:
            dic = {}
            dic['from'] = str(df.loc[i,from_])
            dic['rel'] = rel
            dic['to'] = str(df.loc[i,to])
            kg.append(dic)

        if len(df.loc[i,'grape']) > 0:
            for grape in df.loc[i,'grape']:
                dic = {'from':str(df.loc[i,'item_id']),
                        'rel':'made_from',
                        'to':grape}
                kg.append(dic)
        if len(df.loc[i,'pairing']) >0:
            for pairing in df.loc[i,'pairing']:
                dic = {'from':pairing,
                        'rel':'go_well',
                        'to':str(df.loc[i,'item_id'])}
                kg.append(dic)
                
        for note in notes:
            note_count = wine.loc[i, note+'_count']
            if note_count > 0:
                dic = {'from':str(df.loc[i,'item_id']),
                       'rel':'taste_like',
                       'to':note}
                kg.append(dic)
                for child_note in wine.loc[i, note+'_child'].keys():
                    dic = {'from':child_note,
                           'rel':'part_of',
                           'to':note}
                    kg.append(dic)
    kg = pd.DataFrame(kg)
    kg = kg.dropna()
    return kg


def pykeen_main(args):

    kg = to_kg()
    kg = kg.dropna()
    for col in kg.columns: kg[col] = kg[col].astype(str)

    tf = TriplesFactory.from_labeled_triples(
        kg[["from", "rel", "to"]].values
    )
    training, testing, validation = tf.split([.8, .1, .1])

    if cuda.is_available(): device = 'gpu'
    if args.model == 'TransformerInteraction':
        model_kwargs = dict(
            input_dim=args.embed_dim,
            num_layers = args.num_layer,
            num_heads = args.num_heads,
            drop_out = args.drop_out,
            dim_feedforward = args.dim_ff
        )
    elif args.model == 'ConKB':
        model_kwargs = dict(
            embedding_dim=args.embed_dim,
            num_filters = args.n_filters,
            hidden_dropout_rate = args.drop_out,
        )
    elif args.model == 'RGCN':
        model_kwargs = dict(
            embedding_dim=args.embed_dim,
            num_layers = args.num_layers,
            edge_dropout = args.drop_out,
            self_loop_dropout= args.self_loop_dropout
        )
        args.loss = 'CrossEntropyLoss'
    else:
        model_kwargs = dict(
            embedding_dim=args.embed_dim
        )

    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model=args.model,
        stopper='early',
        loss = args.loss,
        model_kwargs=model_kwargs,
        optimizer_kwargs=dict(lr=args.lr),
        training_kwargs=dict(num_epochs=args.epochs),
        random_seed=args.seed,
        device = device
    )


    return result

def main(args):
    kg = pd.read_csv('/home/dhkim/server_front/winery_AI/winery/wine_kg.csv')
    kg = kg.dropna()
    for col in kg.columns: kg[col] = kg[col].astype(str)

    embed_dim = args.embed_dim
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    margin = args.margin

    torchKGData = KnowledgeGraph(df=kg)

    n_entities=torchKGData.n_ent
    n_relations=torchKGData.n_rel

    model = load_models(args,n_entities, n_relations)
    criterion = MarginLoss(margin)

    if cuda.is_available():
        cuda.empty_cache()
        model.cuda()
        criterion.cuda()
    
    optimizer = Adam(model.parameters(), lr=lr)
    sampler = BernoulliNegativeSampler(torchKGData)
    dataloader = DataLoader(torchKGData, batch_size=args.batch_size, use_cuda="batch")

    iterator = tqdm(range(epochs), unit='epoch')

    for epoch in iterator:
        runningLoss = 0.0
        for batch in dataloader:
            head, tail, relation = batch[0], batch[1], batch[2]
            numHead, numTail = sampler.corrupt_batch(head, tail, relation)
            optimizer.zero_grad()
            pos, neg = model(head, tail, relation, numHead, numTail)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
        iterator.set_description('Epoch %d, loss %.5f' % (epoch, runningLoss/len(dataloader)))

    return model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='pykeen', type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--embed_dim", default=256, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--ent_embed_dim", default=256, type=int)
    parser.add_argument("--rel_embed_dim", default=128, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--n_filters", default=128, type=int)
    parser.add_argument("--lr", default=0.0004, type=float)
    parser.add_argument("--scalar_share", default=0.5, type=float)
    parser.add_argument("--self_loop_dropout", default=0.2, type=float)
    parser.add_argument("--drop_out", default=0.4, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--margin", default=0.5, type=float)
    parser.add_argument("--model", default='ComplEx', type=str)
    parser.add_argument("--loss", default='margin', type=str)
    args = parser.parse_args()

    if args.mode == 'pykeen':
        result = pykeen_main(args)
    elif args.mode == 'torchkge':
        result = main(args)
    pdb.set_trace()
    entity_embedding = result.model.entity_representations[0]._embeddings.weight.data.cpu().numpy()
    relation_embedding = result.model.relation_representations[0]._embeddings.weight.data.cpu().numpy()
    np.save('/home/dhkim/server_front/winery_AI/winery/EDA/graph/embedding/entity_embedding.npy', entity_embedding)
    relation_to_id = result.training.relation_to_id
    entity_to_id = result.training.entity_to_id
    with open('/home/dhkim/server_front/winery_AI/winery/EDA/graph/embedding/relation2idx.json','w') as f: 
        json.dump(relation_to_id,f)
    with open('/home/dhkim/server_front/winery_AI/winery/EDA/graph/embedding/entity2idx.json','w') as f: 
        json.dump(entity_to_id,f)