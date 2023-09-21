# Import necessary modules
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph

# Load environment variables from .env file
load_dotenv()

# Extract OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 1: Establish a connection to the Neo4j graph
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="pleaseletmein"
)

# Step 2: Seed the database with movie data
graph.query(
    """
    MERGE (m:Movie {name:"Top Gun"})
    WITH m
    UNWIND ["Tom Cruise", "Val Kilmer", "Anthony Edwards", "Meg Ryan"] AS actor
    MERGE (a:Actor {name:actor})
    MERGE (a)-[:ACTED_IN]->(m)
    """
)

# Step 3: Refresh and display the graph schema
graph.refresh_schema()
print(graph.get_schema)

# Step 4: Query the graph for actors in "Top Gun"
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
    graph=graph,
    verbose=True
)
chain.run("Who played in Top Gun?")

# Step 5: Limit the results and query again
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
    graph=graph,
    verbose=True,
    top_k=2
)
chain.run("Who played in Top Gun?")

# Step 6: Retrieve and display intermediate results
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
    graph=graph,
    verbose=True,
    return_intermediate_steps=True
)
result = chain.run("Who played in Top Gun?")
print(f"Final answer: {result}")
