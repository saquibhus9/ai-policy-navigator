uploaded_pdf_text = ""
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from retriever import get_similar_chunks, save_pdf_and_index
from llm_answer import ask_mistral
from fastapi.responses import JSONResponse


uploaded_pdf_text = "" 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/ask")
async def ask_question(request: Request):
    global uploaded_pdf_text
    body = await request.json()
    question = body.get("question")

    if uploaded_pdf_text:  # If a PDF has been uploaded
        context = get_similar_chunks(question, uploaded_pdf_text)
        if not context:
            return {"answer": "The uploaded document does not contain relevant information."}
    else:
        context = ""  # No PDF uploaded, fallback to general mode

    answer = ask_mistral(question, context)
    return {"answer": answer}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # ... your indexing logic here ...
        return JSONResponse(content={"message": f"✅ {file.filename} uploaded and processed."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
