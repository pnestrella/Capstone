from scipy.spatial.distance import cosine
from typing import Dict, List
from light_embed import TextEmbedding


# Mode used for the text Embeddings
model = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')


async def reranking(generalSearch : List, userProfile : Dict) -> List:
    #Jobseeker weights
    W1_profile =  userProfile['industry']+". " +", " .join(userProfile['skills']) 
    W2_profile = userProfile['profileSummary'] 

    #Encoding it to vectors
    W1_profileVec = model.encode(W1_profile)
    W2_profileVec = model.encode(W2_profile)


    reranking_output1 = []
    for job in generalSearch:
        #W1 & W2 Job Postings 
        W1_jobposting= ("Industry: " + job['metadata']['industry']+ '.Skills:'+ ', '.join(job["metadata"]['skills']))
        W2_jobposting=(job['metadata']['title']+ '.' + " Industry: " + job['metadata']['industry']+ '. Skills:'+ ', '.join(job["metadata"]['skills']))

        #Encode the W1 & W2 to vectors
        W1_jpVec = model.encode(W1_jobposting)
        W2_jpVec = model.encode(W2_jobposting)

        #Get the cosine similarity
        W1_output =  1- cosine(W1_profileVec,W1_jpVec)
        W2_output = 1-cosine(W2_profileVec, W2_jpVec)

        W1_output = 1 - cosine(W1_profileVec, W1_jpVec)
        W2_output = 1 - cosine(W2_profileVec, W2_jpVec)

        weights = 0.20 * W2_output + 0.80 * W1_output
        #store the outputs
        reranking_output1.append({"id":job["id"], "weights:": {"w1": round(W1_output.tolist(),2), "w2":round(W2_output.tolist(),2)}, "jobDesc": W2_jobposting, "score":round(weights.tolist(),2)})
        
    return reranking_output1