#install the dependecies which i have already done for the previous script.
#I had to pip install the chromadb so i can make a vector database 
import os
from llmware.library import Library
from llmware.retrieval import Query
from llmware.setup import Setup
from llmware.status import Status
from llmware.models import ModelCatalog
from llmware.configs import LLMWareConfig

from importlib import util
if not util.find_spec("chromadb"):
    print("\nto run this example with chromadb, you need to install the chromadb python sdk:  pip3 install chromadb")


def setup_library(library_name):

    

    #Step 1 - Create library which is the main 'organizing construct' in llmware
    print ("\nupdate: Creating library: {}".format(library_name))

    library = Library().create_new_library(library_name)

    #check the embedding status 'before' installing the embedding
    embedding_record = library.get_embedding_status()
    print("embedding record - before embedding ", embedding_record)

    print ("update: Downloading Sample Files")
    #if i want to refresh the sample file folder in case of a crash i can chance over_write to true
    sample_files_path = Setup().load_sample_files(over_write=False)

    #The add.files makes sure the new files are added to the files
    print("update: Parsing and Text Indexing Files")

    library.add_files(input_folder_path=os.path.join(sample_files_path, "Agreements"),
                      chunk_size=400, max_chunk_size=600, smart_chunking=1)

    return library


def install_vector_embeddings(library, embedding_model_name):

    
    #teel the code we need a library object and a vector databse 
    library_name = library.library_name
    vector_db = LLMWareConfig().get_vector_db()

    print(f"\nupdate: Starting the Embedding: "
          f"library - {library_name} - "
          f"vector_db - {vector_db} - "
          f"model - {embedding_model_name}")

    #this line of code is used to create the embeddings in which i specify the model and what databse to use
    #You can also specify the batch size to tweak the performance which i could have done better
    library.install_new_embedding(embedding_model_name=embedding_model, vector_db=vector_db,batch_size=100)

    #The code keeps updating and printing the results so you can track your progress.
    update = Status().get_embedding_status(library_name, embedding_model)
    print("update: Embeddings Complete - Status() check at end of embedding - ", update)

    # Start using the new vector embeddings with Query
    sample_query = "how much pay will i recieve?"
    print("\n\nupdate: Run a sample semantic/vector query: {}".format(sample_query))

    #queries are constructed by creating a Query object, and passing a library as input
    query_results = Query(library).semantic_query(sample_query, result_count=20)

    for i, entries in enumerate(query_results):

        #each query result is a dictionary with many useful keys listed below
        text = entries["text"]
        document_source = entries["file_source"]
        page_num = entries["page_num"]
        vector_distance = entries["distance"]


        #for display purposes i will only display the first 125 characters of the text to keep the terminal a bit cleaner
        if len(text) > 125:  text = text[0:125] + " ... "

        print("\nupdate: query results - {} - document - {} - page num - {} - distance - {} "
              .format( i, document_source, page_num, vector_distance))

        print("update: text sample - ", text)

    #lets take a look at the library embedding status again at the end to confirm embeddings were created
    embedding_record = library.get_embedding_status()

    print("\nupdate:  embedding record - ", embedding_record)

    return 0


if __name__ == "__main__":

    #set up the collection databse using SQLite

    LLMWareConfig().set_active_db("sqlite")

    #Set up the vector data base using chromadb
    LLMWareConfig().set_vector_db("chromadb")

    #Using the libary i made for my last coding project
    library = Library().load_library("example1_library")


    #line of code to be able to pick an embedding model

    embedding_models = ModelCatalog().list_embedding_models()

    #for this code i chose mini-lm-sbert because this a a small and fast embedding model recommended in the turtorial i followed 
    embedding_model = "mini-lm-sbert"

    #run the core script
    install_vector_embeddings(library, embedding_model)


