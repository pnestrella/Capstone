from sentence_transformers import SentenceTransformer
from typing import Dict, List
from pinecone import Pinecone
#setting up dotenv
import os

#Model for VECTORIZING
model = SentenceTransformer('all-MiniLM-L6-v2')

#This is for pinecone / GENERAL Searching
async def pineconeSearch(userQuery : Dict) -> List:
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("job-listings")
      
    #User Search
    userSearch = userQuery['userSearch']['query']
    searchIndustry = userQuery['userSearch']['queryIndustry']
    
    #Vectorize The Search
    userSearchVectorized = model.encode(userSearch).tolist()

    #Pass the Vectorize search to Pinecone for better searching
    generalSearchResult = index.query(
    vector=userSearchVectorized, 
    top_k=15,
    include_metadata=True,
    filter={
        "jobUID": {"$nin": userQuery['skippedJobs']},
        "industry" : searchIndustry,
        # "job_id" : {"$in": ["job1","job6","job11","job18"]}
    }
    )
    generalSearchResult = generalSearchResult['matches']

    #Threshold for the GENERAL Searching filter; for those with xx%//maintaining accurate results
    filteredList = [i for i in generalSearchResult if i.get('score',0) > 0.30]
    return filteredList