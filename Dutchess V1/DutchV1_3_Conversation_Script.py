# %% [markdown]
# This Jupyter Notebook goes over the process of creating a starting a conversation with the chatbot, please make sure to use the "Resource Creator" and the "Assistant Creator" before you run this script.
# 
# In this notebook we retrieve all the files we created in the previous notebook and use them to create a conversation with the chatbot.
# 
# # Contents
# 
# **1 - Setup**
# 
# - 1.1 Imports
# - 1.2 OpenAI
# - 1.3 Directories
# - 1.4 Previous Files
# 
# **2 - Knowledge Retrieval Functions**
# 
# - 2.1 Query Embedding and Comparison
# - 2.2 Chunking and Specified Knowledge Retrieval
# - 2.3 Image Retrieval
# 
# **3 - Conversation**
# 
# - 3.1 Event Handler
# - 3.2 Conversation Handler

# %% [markdown]
# # 1 - Setup
# 
# This section details all of the basic peices we need to put together for our chatbot to function, it does not include the functions we use for the chatbot but does detail the imports, OpenAI functionality, directories and the loading of the files we set up in the "Assistant Creator" and "Resources Creator".

# %% [markdown]
# ## 1.1 Imports
# 
# These are our imports for this notebook, I'll go over what each are used for now:
# 
# `from openai import OpenAI` the OpenAI module allows us to set up a client that can communicate with OpenAI's services. These services are not specific to just chatbots although it does include this purpose we can use these services to create vector stores (more on this later) and upload and change files.
# 
# `import os` this module allows us to modify and access files and folders
# 
# `import json` this module allows for the reading and creation of .json files which allow us to store the data we process for later use
# 
# `import requests` this module allows us to make external requests to outside urls, specifically we will be making requests to OpenAI
# 
# `from PIL import Image` this module allows for the storage and retreival of images given image data
# 
# `import pickle` this module allows us to store what cannot be in json files due to information being non-subscriptable
# 
# `import numpy as np` this module is for math process'

# %%
from openai import OpenAI
import os
import json
import requests
from PIL import Image
import pickle
import numpy as np

# %% [markdown]
# If any errors are returned when trying to run the above due to modules not being installed you can remove the # from the appropriate commands below to install the module.

# %%
#pip install openai
#pip install os
#pip install json
#pip install requests
#pip install PIL
#pip install pickle
#pip install numpy

# %% [markdown]
# ## 1.2 OpenAI
# 
# We define the defnitions needed for OpenAI so we can easily access them later
# 
# `api_key =` this is essentially a password provided by OpenAI, it allows us to access OpenAI's services whenever we use them
# 
# `client = OpenAI(api_key=api_key)` this sets up a client which can communicate with OpenAI's services, we specify this beforehand so we do not have to write out "OpenAI(api_key=api_key)" when we want to communicate with OpenAI

# %%
api_key = ""

client = OpenAI(api_key=api_key)

# %% [markdown]
# ## 1.3 Directories
# 
# We set up any directories for files that we will use later:
# 
# `store_name =` this is a general purpose name that we will use when creating files, this allows us to make sure we are retrieving the documents we want later on.
# 
# `data_directory =` this is the file directory where we'll store and retrieve any our data.
# 
# `document_directory =` this is the file directory where we'll store and retrieve our documents from.
# 
# `image_directory =` this is the file directory where we'll store and retreieve any images.
# 
# `assistant_directory =` this is the file directory where we store and retrieve the assistant ids from. 
# 
# You should make sure when specifying these that they are the same as you used in the Assistant Creator and Resource Creator Scripts

# %%
store_name = "Labs Dutchess"

this_directory = os.getcwd()

directories = os.listdir(this_directory)

directories = [os.path.join(this_directory, entry) for entry in directories if not os.path.isfile(os.path.join(this_directory, entry))] 

for directory in directories:
    if "Data Base" in os.path.basename(directory):
        data_directory = directory
    elif "Documents" in os.path.basename(directory):
        document_directory = directory
    elif "Output Images" in os.path.basename(directory):
        image_directory = directory
    elif "Assistants" in os.path.basename(directory):
        assistant_directory = directory

print(f"data_directory = {data_directory}")
print(f"document_directory = {document_directory}")
print(f"image_directory = {image_directory}")
print(f"assistant_directory = {assistant_directory}")

# %% [markdown]
# ## 1.4 Previous Files
# 
# We need retrieve all of the items we stored, for the very last one of these make sure you've changed the "assistant_name" manually in the box above.

# %%
allias_name = f"allias_{store_name}.json" 
allias_path = os.path.join(data_directory, allias_name) # gets the path for our allias'

with open(allias_path, "r") as file:
    allias = json.load(file) # retrieves our allias'

#-------------------------------------------------------------------------------------------------

image_names_list = f"{store_name}_image_names.json"
image_names_path = os.path.join(data_directory, image_names_list) # gets the path for our image names

with open(image_names_path, "r") as file:
    image_names = json.load(file) # retrieves the image names

#-------------------------------------------------------------------------------------------------

descriptions_name = f"{store_name}_descriptions.json" 
descriptions_path = os.path.join(data_directory, descriptions_name) # gets the path for our descriptions

with open(descriptions_path, "r") as file:
    descriptions = json.load(file) # retrieves our descriptions

#-------------------------------------------------------------------------------------------------

chunks_name = f"chunks_{store_name}.json"
chunks_path = os.path.join(data_directory, chunks_name) # gets the paths for our chunks

with open(chunks_path, "r") as file:
    chunks = json.load(file) # retrieves our chunks

#-------------------------------------------------------------------------------------------------

data_path = os.path.join(data_directory, f"{store_name}database.pkl") # gets the path for the embedding file

with open(data_path, 'rb') as f:
    db = pickle.load(f) # retrieves our embedding

#-------------------------------------------------------------------------------------------------

if os.listdir(document_directory) != []:
    vector_name = f"vector_store_id_{store_name}.json"
    vector_path = os.path.join(data_directory, vector_name) # gets the path for our vector store id

    with open(vector_path, "r") as file: 
        vector_store_id = json.load(file) # retrieves our vector store id

# %% [markdown]
# # 2 - Knowledge Retrieval Functions
# 
# This section covers the basic functions our chatbot will need to ensure it is retrieving the knowledge most relevant to the user query.

# %% [markdown]
# ## 2.1 Query Embedding and Comparison
# 
# We start off by embedding our querry as we did for the chunks previously, this allows us to compare the similarity of the query to each of the different chunks

# %%
def query_embedd(query):    
    query_response = client.embeddings.create( #creates an embedding
        model="text-embedding-ada-002", # picks a model to use for embedding, this is a general purpose one for text but there are others for other purposes
        input=[query], # selects what list we want to use for our embedding
        encoding_format="float" # selects what format the embedding is in, the other option is base64
    )

    query_embedding1 = query_response.data[0].embedding 
    query_embedding2 = np.array(query_embedding1).flatten() # we turn our query embedding into a single flattened vector 
    return query_embedding2 # we return the flattened embedding

# %% [markdown]
# We create a function that compares the similarity of two vectors using the angle between them as a measure, our two vectors will be the embedding of the query and each of the different chunks

# %%
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) # formula for the cosine of an angle between two vectors


# %% [markdown]
# We then define a function that takes the embedding of a chunk and uses the cosine_similarity function to compare it against the embedding of our query. We store all of the output values in a list

# %%
def get_similarities(query_embedding):
    similarity_scores = [] # creates a list of similarity scores
    for embedding_data in db.data: # loops through the data assisgned to each chunk
        chunk_embedding = embedding_data.embedding # retrieves the embedding from a specific chunks embedding

        chunk_embedding = np.array(chunk_embedding).flatten() # turns our embedding into a single flattened vector

        score = cosine_similarity(query_embedding, chunk_embedding) # compares the similarity of the two vectors
        similarity_scores.append(score) # store the similarity of the two vectors
    return similarity_scores # returns the list of similarity scores between the query and the chunks

# %% [markdown]
# ## 2.2 Chunking and Specified Knowledge Retrieval
# 
# We now define a series of functions that allow us to retrieve the most relevant chunks to our given query
# 
# This is function retrieves a list of all the files in a given folder path. We'll use these folder paths as a way of referencing which document the chunks we end up using come from.

# %%
def get_all_files_in_folder(folder_path):
    try:
        entries = os.listdir(folder_path) # Makes a list of all entries in a given directory
        
        files = [os.path.join(folder_path, entry) for entry in entries if os.path.isfile(os.path.join(folder_path, entry))] # combines the entries with the folder directory so that we have their full file path
        
        return files
    except FileNotFoundError:
        return "The folder path does not exist."

# %% [markdown]
# We then define a function that takes the similarity scores and finds the top chunks that are most similar to the user query and outputs these top chunks into a single string, it checks the start of each of the chunks for where they are sourced from and adds these sources to a list of sources. It uses the allias' defined in the reasource creator so that when we give the sources to the user they are more readable. If no allias are given then the source is simply the file name.

# %%
def get_combined_string(similarity_scores):
    files = get_all_files_in_folder(document_directory)
    chunk_scores = list(zip(chunks, similarity_scores)) # combines the chunks and similarity scores into a list with each chunk indexed against its similarity score
    sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True) # sorts the list according to the similarity scores

    top_n = 5 # decides how many chunks to be accepted into the string
    combined_string = f""

    all_sources =[] # empty string for sources of each chunk
    sources = [] # empty string for non duplicate sources

    for i in range(top_n):
        combined_string += sorted_chunks[i][0] # combines the top chunks into a single string

        for file in files:  # Loop through all the file names
            base_filename = os.path.basename(file)
            chunk_prefix = sorted_chunks[i][0]
            
            if chunk_prefix.startswith(base_filename):  # Check the chunk to see which document it is sourced from
                found = False
                for n in range(len(allias)):
                    if base_filename == allias[n][0]:
                        all_sources.append(allias[n][1])  # Add the source to a list
                        found = True
                        break  # Exit the inner 'for n' loop
                if not found:  # This runs only if no alias match was found
                    all_sources.append(base_filename) # adds the file name as the source
                break  # Exit the outer 'for file' loop to avoid unnecessary iterations

    sources = list(set(all_sources)) # removes duplicates from the list
    
    return combined_string, sources # returns our string and the sources of each of the chunks

# %% [markdown]
# ## Image Retrieval
# 
# Given a user query we also want to retrieve an image most relevant to the user query to do this we create a prompt to send to gpt-4o: 
# `out of this list {descriptions} which relates most to {query}, if none of them relate to the query state only the word false otherwise state only a single number starting with the first being 0 as this is being used for indexing in python`,
#  this prompt asks gpt-4o which out of the descriptions of the images which are most relevant to the query and if none are it returns `false`. We can then use this output to decide which image to show to the user.

# %%
def image_retrieval(query):  
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = f"Out of this list {descriptions} which relates most to {query}, if none of them relate to the query state only the word false otherwise state only a single number starting with the first being 0 as this is being used for indexing in python, please do not be afraid to use the word false, you should only return indexes on average 30% of the time."
    message = {  # this is our "message" it's what we're actually sending to OpenAI
        "model": "gpt-4o",  # model to send the prompt to
        "messages": [  # contains the main content of what we want to send
            {
                "role": "user",  # specifies that this is a message from a user
                "content": [  # what is included in our message, what gpt-4o will see
                    {
                        "type": "text",  # specifies what the following input is
                        "text": prompt  # we tell gpt-4o what we want it to do
                    },
                ]
            }
        ],
        "max_tokens": 300  # sets a limit on the number of tokens to be used per message, tokens equate to processing which equates to money, so by limiting this we keep cost and processing time down
        # note that increases in the max_tokens is necessary for more complex tasks
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=message)  # we post the message to chat/completions
    response_data = response.json()
    index = response_data['choices'][0]['message']['content'].strip()

    if "a" in index or "A" in index: # checks if there is a relavent image
        pass
    else:
        index = int(index)
        image_name = image_names[index]
        image_path = os.path.join(image_directory, image_name)
        description = descriptions[index]
        return Image.open(image_path), description, image_path, str(image_name)

# %% [markdown]
# # 3 - Conversation
# 
# This section details the Event Handler and functions that are run when we have a conversation with the chatbot. These functions contain most if not all of the functions we have defined previously.

# %% [markdown]
# ## 3.1 Event Handler
# 
# This is our event handler class, it essentially handles how our output looks, it handles what we as a user see. This event handler is able to retrieve images that our chatbot creates such as graphs it generates through its code_interpreter and it displays any text the chatbot writes in a markdown box which provides a readable form, it also makes it so that the text is updated as the chatbot writes it.

# %%
from IPython.display import display, Markdown, clear_output # imports that handle the display
import requests # allows us to retrieve images from OpenAI
from PIL import Image # allows us to display those images
import io # allows us convert images to the relevant format
from openai import AssistantEventHandler # allows us to create an event handler
from typing_extensions import override # allows us to use override

class EventHandler(AssistantEventHandler):

    def __init__(self, sources):
        super().__init__()
        self.buffer = ""  # Buffer to collect text output
        self.sources = sources  # List to collect sources
        self.final_output = ""  # Variable to store final output

    @override # @override is a decorator used to explicitly indicate that this method is overriding a method in the superclass
    def on_text_created(self, text):
        self.update_buffer(text.value) # stores and updates the buffer text with the initial text

    @override
    def on_text_delta(self, delta, snapshot):
        self.update_buffer(delta.value) # stores and updates the buffer text with new text as the chatbot writes 

    @override
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True) # tells us when the chatbot uses a tool 
            
    def on_tool_call_delta(self, delta, snapshot): # updates as a tool is used, specifically for the code interpreter to show images that it creates
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True) # tells us that the code interpreter is giving an output
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True) # prints the logs of the code interpreter
                    elif output.type == "image":
                        file_id = output.image.file_id
                        image_data = self.download_image(file_id) # downloads the image from OpenAI using the download image function
                        if image_data:
                            image = Image.open(io.BytesIO(image_data))
                            image.show() # shows the downloaded image to the user

    def download_image(self, file_id):
        url = f"https://api.openai.com/v1/files/{file_id}/content" # the url for where assistant generated outputs go
        headers = {
            "Authorization": f"Bearer {api_key}", # our api key in our header
        }
        response = requests.get(url, headers=headers) # gets the response from OpenAI
        if response.status_code == 200:
            return response.content # returns the image
        else:
            print(f"Failed to download image: {response.status_code} {response.text}")
            return None
        
    def update_buffer(self, text): # this displays the output text and updates it as the chatbot writes more of the output
        if not self.buffer.endswith(text):  # Prevent duplication
            self.buffer += text
            self.display_output() # displays the output

    def display_output(self):
        clear_output(wait=True) # Clear previous output
        processed_content = self.format_buffer(self.buffer) # Process the buffer to format LaTeX and code blocks
        processed_content += self.format_sources()  # Append sources information
        self.final_output = processed_content  # Store final output
        display(Markdown(processed_content)) # Display as Markdown to correctly render LaTeX and plain text

    def format_buffer(self, buffer):
       # Split buffer into lines for processing
        lines = buffer.split("\n")
        formatted_lines = []

        in_code_block = False

        for line in lines:
            # checks to see if the lines are a code block
            if line.strip().startswith("code_interpreter"):
                in_code_block = True
                formatted_lines.append(line)
            elif line.strip() == "```":
                in_code_block = False
                formatted_lines.append(line)
            elif in_code_block:
                formatted_lines.append(line)
            else:
                # Check for LaTeX patterns and wrap them in delimiters
                line = line.replace(r'\(', '$').replace(r'\)', '$')
                line = line.replace(r'\[', '$$').replace(r'\]', '$$')
                formatted_lines.append(line)
        
        return "\n".join(formatted_lines) # joins all of the lines together

    def format_sources(self):
        if self.sources:
            return f"\n\nAll information has been sourced from {', '.join(self.sources)}" #adds this line onto the end of all outputs so that sources are always given
        return ""

    def get_final_output(self):
        return self.final_output  # Return the final processed content

# %% [markdown]
# ## 3.2 Conversation Handler
# 
# Our Conversation Handler consists of 2 functions the `start_conversation` function and the `continue_conversation` function, both work in very similar ways. 
# 
# **`start_conversation` function**
# This function takes in several arguments, these are: 
# - `prompt` this should be a processed peice of text that includes the original user query, any instructions on how the assistant should process the user query and any background information we want to give the assistant without it having to perform a file search. 
# - `assistant_name` is the name of the assistant to be used in response to the prompt these should match those created in `DutchV1_3_Assistant_Creator`
# - `sources` this should contain a list of sources that the background information is based upon
# - `image_data` should contain any image_data that needs to be shown to the user
# This function then starts by extracting the assistant id based on the assistant name entered, it then checks if there is image_data to be processed, if there is the process of giving the prompt to the assistant includes said image however if there is not then the prompt is given to the assistant without an image. We then retrieve the response and if there was an image we append its name and description to the response. The function then returns as its output:
# - the `thread` so we can continue conversation after the function
# - the final text output so we can show the user 
# and if there was an image:
# - the file reference uploaded to OpenAI so we can delete it from their servers once the conversation is finished
# 
# **`continue_conversation` function**
# This function is very similar to the `start_conversation` function, the only noticible difference being the lack of image processing and it taking the `thread` as an input. The reason we do not do image processing in this prompt is that we do not want every follow up message in the conversation to contain an image we only want certain follow up prompts to be met with an image such as "show me an image of" prompts, for this we assign the image processing to the `continue_conversation_with_image` function that does the exact same thing as the `continue_conversation` function but with image processing. 
# 
# **`manual_assistant_message_start` and `manual_assistant_message_continue` functions**
# These functions are extremely simple and allow us to manually add assistant messages to the thread. They could be called, for example, in instances of users asking silly questions that we want to give a fixed response of "I am afraid I cannot answer that, I'd be happy to assist you with any other queries you have" to. The `assistant_output` is the argument into which the fixed response should be entered as a string and for the continue function the thread also needs to be passed so that the messages is added to the correct conversation.

# %%
def start_conversation(prompt, assistant_name, sources, image_data): 
  
  assistant_name = f"{assistant_name}_assistant_id.json" 
  assistant_path = os.path.join(assistant_directory, assistant_name) # gets the path for our vector store id

  with open(assistant_path, "r") as file: 
    assistant_id = json.load(file) # retrieves our vector store id


  if image_data != None:
    image, description, image_path, image_name = image_data
    file = client.files.create(file=open(image_path, "rb"), purpose="vision")
    thread = client.beta.threads.create(  # we create a thread (conversation)
        messages=[  # what we want to send to the assistant
            {
                "role": "user",  # who we are sending the message from
                "content": 
                [
                    {
                        "type": "text",
                        "text": prompt,  # what our message is
                    },
                    {
                    "type": "image_file",
                    "image_file": {"file_id": file.id},
                    },
                    
                ]
            },
        ]
    )

    event_handler = EventHandler(sources=sources)  # we define the event handler which we want to use

    with client.beta.threads.runs.stream(  # we send our thread to an assistant and specify what event handler to use
        thread_id=thread.id,  # what thread we want to use
        assistant_id=assistant_id,  # what assistant we want to use
        event_handler=event_handler,  # what event handler we want to use
    ) as stream:
        stream.until_done()  # streams our output

    event_handler.display_output()  # displays our output

    
    output = f"{event_handler.get_final_output()}"

    image, description, image_path, image_name = image_data
    image.show()  # an image is shown if there is a relevant image 
    output += f"\n\n *Image Description of {image_name}*: {description}" # append the description of the image to the text

    output += f"\n\n *Thread ID*: {thread.id}"
    return thread, output, file

  else:
     thread = client.beta.threads.create(  # we create a thread (conversation)
        messages=[  # what we want to send to the assistant
            {
                "role": "user",  # who we are sending the message from
                "content": prompt,
            },
        ]
    )
     event_handler = EventHandler(sources=sources)  # we define the event handler which we want to use

     with client.beta.threads.runs.stream(  # we send our thread to an assistant and specify what event handler to use
        thread_id=thread.id,  # what thread we want to use
        assistant_id=assistant_id,  # what assistant we want to use
        event_handler=event_handler,  # what event handler we want to use
    ) as stream:
        stream.until_done()  # streams our output

     event_handler.display_output()  # displays our output

    
     output = f"{event_handler.get_final_output()}"

     output += f"\n\n *Thread ID*: {thread.id}"
     return thread, output, None

     

# %%
def continue_conversation(prompt, thread, assistant_name, sources):

  assistant_name = f"{assistant_name}_assistant_id.json" 
  assistant_path = os.path.join(assistant_directory, assistant_name) # gets the path for our vector store id

  with open(assistant_path, "r") as file: 
    assistant_id = json.load(file) # retrieves our vector store id
 
# the prompt to send to the user
  thread_message = client.beta.threads.messages.create(
      thread_id=thread.id,
      role="user",
      content=prompt
  )

  event_handler2 = EventHandler(sources=sources)

  with client.beta.threads.runs.stream(
    thread_id=thread.id,
    assistant_id=assistant_id,
    instructions="",
    event_handler=event_handler2,
  ) as stream:
    stream.until_done()

  event_handler2.display_output()

  return event_handler2.get_final_output()

# %%
def continue_conversation_with_image(prompt,thread, assistant_name, sources, image_data):
  image, description, image_path, image_name = image_data

  assistant_name = f"{assistant_name}_assistant_id.json" 
  assistant_path = os.path.join(assistant_directory, assistant_name) # gets the path for our vector store id

  with open(assistant_path, "r") as file: 
    assistant_id = json.load(file) # retrieves our vector store id
 
# the prompt to send to the user
  file = client.files.create(file=open(image_path, "rb"), purpose="vision")
  thread_message = client.beta.threads.messages.create(
      thread_id=thread.id,
      role = "user",  # who we are sending the message from
      content =[{"type": "text","text": prompt,},{"type": "image_file","image_file": {"file_id": file.id},},]
    )

  event_handler2 = EventHandler(sources=sources)

  with client.beta.threads.runs.stream(
    thread_id=thread.id,
    assistant_id=assistant_id,
    instructions="",
    event_handler=event_handler2,
  ) as stream:
    stream.until_done()

  event_handler2.display_output()

  output = f"{event_handler2.get_final_output()}"
  image.show()  # an image is shown if there is a relevant image 
  output += f"\n\n *Image Description of {image_name}*: {description}" # append the description of the image to the text

  return output,file

# %%
def manual_assistant_message_start(assistant_output):

    thread = client.beta.threads.create(  # we create a thread (conversation)
        messages=[  # what we want to send to the assistant
            {
                "role": "assistant",  # who we are sending the message from
                "content": assistant_output,
            },
        ]
    )
    return thread, None

# %%
def manual_assistant_message_continue(assistant_ouput, thread,):
        
  thread_message = client.beta.threads.messages.create(
      thread_id=thread.id,
      role="assistant",
      content=assistant_ouput
    )

# %% [markdown]
# Within the main folder you should also find a .py copy of this notebook, this copy enables DutchessV1 to call the functions within this script. If there is not a .py copy please create one by clicking the three dots at the top and exporting as a python script. The python script should have the name: `DutchV1_3_Conversation_Script.py`

# %% [markdown]
# 


