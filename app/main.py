from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from dotenv import load_dotenv

from app.pipelines.generation import generation
from app.pipelines.pineconeSearch import pineconeSearch
from app.pipelines.reranking import reranking

load_dotenv()

app = FastAPI()

class RecommendationRequest(BaseModel):
    profileQuery: Dict 

@app.get('/')
async def root():
    return ({"message": "fast api is running :)"})

#To get the recommendation request from models
@app.post('/api/getReco')
async def getRecommendation(req: RecommendationRequest):

    #General search outputs
    generalSearchInputs = {
        "userSearch": req.profileQuery['userSearch'],
        "skippedJobs":req.profileQuery['skippedJobs']
    }
    #Reranking inputs
    rerankingInputs = {
            "skills" : req.profileQuery['skills'],
            "industry": req.profileQuery['industry'],
            "profileSummary":req.profileQuery['profileSummary']
    }

    generationInputs = {
        "profileSummary": req.profileQuery['profileSummary'],
        "industry": req.profileQuery['industry'],
        "skills": req.profileQuery['skills'],
        "experience": req.profileQuery['experience'],
        "certifications":req.profileQuery['certifications']
    }


      
    try:
        #Part (1)
            #(Phase 1) - General Search
        generalSearch = await pineconeSearch(generalSearchInputs)
            #(Phase 2) - Semantic Reranking
        rerankingPhase = await reranking(generalSearch, rerankingInputs)
        # #Part (2)
        #     #(Phase 3) - Feedback from LLM/ Generation
        generationPhase = await generation(generationInputs,rerankingPhase)
 
        return generationPhase

        
    except Exception as err:
        HTTPException(status_code=500, detail=f"There's an error in {str(err)}")




    
