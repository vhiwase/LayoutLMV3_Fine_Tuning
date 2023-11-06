# pip install python-louvain

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MeanShift
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import networkx as nx
import community  # Louvain method
from collections import defaultdict


model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# text_data = new_role['top_left_x'].apply(lambda x: str(x)+ '_') + new_role['top_right_x'].apply(str)
new_role = paragraph_dataframe[420:1420]
new_role = new_role.reset_index(drop=True)
sentences = new_role['text'].tolist()
embeddings = model.encode(sentences)



# # X = new_role[['top_left_x', 'top_right_x']].values
 
# clustering = MeanShift()  # Adjust the number of clusters based on your needs
# clustering.fit(X)
# labels = clustering.labels_

# text_data = new_role['content'].tolist()



# # Assign the cluster labels to the content
# for i, label in enumerate(labels):
#     print(f'Cluster {label}: {text_data[i]}')





def get_faiss_index(embeddings, nn_to_map=2000):
    def create_faiss_index(embeddings,ids,nn_to_map=2000):    
        vector_dimension = embeddings[0].shape[0]    
        cluster_size = nn_to_map    
        nlist = int(np.ceil(len(embeddings) / cluster_size))
        quantizer = faiss.IndexFlatL2(vector_dimension)    
        embeddings = np.vstack(embeddings)    
        ids = np.hstack(ids)    
        faiss.normalize_L2(embeddings)    
        faiss_index = faiss.IndexIVFFlat(quantizer, vector_dimension, nlist)
        if faiss_index.is_trained is False:
            faiss_index.train(embeddings)    
        faiss_index.add_with_ids(embeddings, ids)    
        faiss_index.nprobe = min(10,nlist)
        return faiss_index

    print("source embedding completed")
    ids = list(range(len(embeddings)))
    ids = np.array(ids, dtype=np.int64)
    print("Creating faiss index ...")
    faiss_index = create_faiss_index(embeddings,ids,nn_to_map=nn_to_map)
    return faiss_index


faiss_index = get_faiss_index(embeddings, nn_to_map=4)
print(faiss_index.ntotal)


text_embedding = embeddings[0]
xq = np.array([text_embedding])

D, I = faiss_index.search(xq, k=100)
normalized_score = (4-D)/4
print(I, normalized_score)


new_role['embeddings'] = list(embeddings)

new_role['xq'] = new_role['embeddings'].apply(lambda text_embedding: np.array([text_embedding]))

new_role['D_I'] = new_role['xq'].apply(lambda xq: faiss_index.search(xq, k=20))

new_role['D'] = new_role['D_I'].apply(lambda x: x[0])
new_role['I'] = new_role['D_I'].apply(lambda x: x[1])

new_role['normalized_score'] = new_role['D'].apply(lambda x: (4-x)/4)

d = pd.DataFrame()
d['I'] = new_role['I'].apply(lambda x: x[0])
d['normalized_score'] = new_role['normalized_score'].apply(lambda x: x[0])
d['text'] = new_role['text']
d['index'] = new_role.index

# X = d[["index", "I"]].values

# clustering = MeanShift()  # Adjust the number of clusters based on your needs
# clustering.fit(X)
# labels = clustering.labels_

# d['labels'] = labels



# import networkx as nx
# import matplotlib.pyplot as plt

# # Create a directed graph from the NumPy array
# G = nx.DiGraph()
# for edge in X:
#     G.add_edge(edge[0], edge[1])

# # Visualize the graph
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8, font_color='black', arrowsize=10)
# plt.title("Directed Graph")
# plt.show()






# Create a graph from the edge list
G = nx.Graph()
edge_list = new_role['I'].apply(lambda x: list(x[0][0:2])).tolist()
G.add_edges_from(edge_list)

# Detect communities using the Louvain method
partition = community.best_partition(G)

assoication_dict = defaultdict(list)
# Print the detected communities
for community_id, nodes in partition.items():
    assoication_dict[nodes].append(community_id)
    print(f'Community {community_id}: {nodes}')
