## Network of Agents with AutoGen

<img width="1487" height="591" alt="image" src="https://github.com/user-attachments/assets/5a61ffad-9a45-46f5-8702-0c830844b951" />
# autogen-exercise

# Goal:
  Explore how to create a network of agents (managers and specialists) to answer user query's.

# Note:
  Still with some bug in the front end.

# How to run it
 - Clone the repo
 - create the virtual environment (I used  virtualenv)
 - update the .env file for each agent and the coordinator
 - for each agent, coordinator and registry open a separated terminal:
   - activate the virtual environment and run the python agent.py inside each folder
   - to check if the agent is running, go to the http://localhost:PORT/a2a
 - Run the coordinator as the last, open the provided url and start sending querys.

# How it works:
  - User submit a query
  - Coordinator get the query:
    - Create a network of managers
    - Check the best manager to answer the question
    - Submit the query to manager
    - the Manager break the query into atomic tasks
    - Coordinator instanciate a network of specilist linked to the manager
    - for each task coordinator submit to the specialist
    - Update the user with each response
