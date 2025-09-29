from python_a2a import Message, TextContent, MessageRole
from python_a2a import AgentNetwork, A2AClient, AIAgentRouter, Flow
from python_a2a.client.llm import OpenAIA2AClient

import requests
import os
from dotenv import load_dotenv
import json
import asyncio

#DEBUG
import traceback

#INTERFACE
import gradio as gr

load_dotenv()


#create a agent network from registry
def create_agent_network_from_registry(
        registry_url: str, 
        domain_expertise: str = None, 
        subdomain_expertise: str = None) -> AgentNetwork:
    
    response = requests.get(f"{registry_url}/registry/agents")
    agents = response.json()
    network = AgentNetwork(name="Dynamic Agent Network")

    #Network of specialized Agents based on subdomain_expertise
    if subdomain_expertise:
        for agent in agents:
            if agent["capabilities"]["agent_subdomain_expertise"].lower() == subdomain_expertise.lower():
                network.add(agent['name'], agent['url'])
    
    #Network of Agents with Management Capacities
    elif domain_expertise:
        for agent in agents:
            if agent["capabilities"]["agent_domain_expertise"].lower() == domain_expertise.lower():
                network.add(agent['name'], agent['url'])
    
    #Network with all agents
    else:
        for agent in agents:
            network.add(agent['name'], agent['url'])

    return network

#cria  rede de gerentes
def get_manager_network():
    managers_network = create_agent_network_from_registry(os.getenv("AGENT_REGISTRY_URL"),domain_expertise="management")
    print("Network of managers")
    print(managers_network)
    for agt in managers_network.list_agents():
        print(agt)
    print("^^"*20)

    return managers_network

#recebe a demanda e define com base nos agentes qual e o melhor para responder a demanda
def get_best_manager(network: AgentNetwork, llm_client, query):

    #create the router to the best manager
    router = AIAgentRouter(
        llm_client=llm_client,
        agent_network=network
    )

    #get the best manager to answer the task
    agent_name, confidence = router.route_query(query)
    agt_card = network.get_agent_card(agent_name).to_dict()
    expertise = agt_card["capabilities"]["agent_subdomain_expertise"]

    return network.get_agent(agent_name), expertise, agent_name


def chat_demo(message, history):
    try:

        #Create LLM object
        openai_client = OpenAIA2AClient(
            api_key=os.getenv("AGENT_MODEL_API_KEY"),
            model=os.getenv("AGENT_MODEL_NAME"),
        )

        #get the manager and also the team expertise (subdomain_expertise)
        agent_manager, expertise, agent_name = get_best_manager(
            create_agent_network_from_registry(os.getenv("AGENT_REGISTRY_URL"),domain_expertise="management"), # Managers Network
            openai_client,
            message
        )  
        history.append({
            "role": MessageRole.AGENT,
            "Agent profile": 'data_manager',
            "content": f"Agent selected to answer your question: {agent_name}"
        })
        yield history 
        #yield f"Agent selected to answer your question: {agent_name}"

        #create the second network, with Specialized agents
        network = create_agent_network_from_registry(
            os.getenv("AGENT_REGISTRY_URL"),
            subdomain_expertise=expertise
        )

        #create the specialized Agent router
        router = AIAgentRouter(
            llm_client=openai_client,
            agent_network= network
        )

        #warmup
        query = "Be ready to receive prompts."
        msg = Message(content=TextContent(text=query), role=MessageRole.USER)
        resp = agent_manager.send_message(msg)

        #user query
        query = message
        msg = Message(content=TextContent(text=query), role=MessageRole.USER)
        resp = agent_manager.send_message(msg)
        rsp = json.loads(resp.to_json())
        print(rsp)
        rsp = json.loads(rsp["content"]["text"])
        print("**0"*20)
        print(rsp)
        #verify the agent answer
        if rsp["ok"]:
            if not 'tasks' in rsp["data"]:
                history.append({
                    "role": MessageRole.AGENT,
                    "Agent profile": 'data_manager',
                    "content": rsp["data"]
                })
                yield history
                #yield rsp["data"]
                return

            tasks = json.loads(rsp["data"])
            history.append({
                "role": MessageRole.AGENT,
                "Agent profile": 'data_manager',
                "content": f'To better answer your question I decoupled it in the following tasks and will send each one to a specialized agent to answer: {rsp["data"]}'
            })
            yield history
            #yield f'To better answer your question I decoupled it in the following tasks and will send each one to a specialized agent to answer: {rsp["data"]}'
             

            for task in tasks["tasks"]:
                agent_name, confidence = router.route_query(task)
                print(f"Routing to {agent_name}, confidence of {confidence}")
                #ield f"task: {task} sent to {agent_name}"
                history.append({
                    "role": MessageRole.AGENT,
                    "Agent profile": {agent_name},
                    "content": f"task: {task} sent to {agent_name}"
                })
                yield history

                msg = Message(content=TextContent(text=task), role=MessageRole.USER)
                rsp_agt = network.get_agent(agent_name).send_message(msg)
                rsp = json.loads(rsp_agt.to_json())
                rsp = json.loads(rsp["content"]["text"])
                history.append({
                    "role": MessageRole.AGENT,
                    "Agent profile": {agent_name},
                    "content": rsp["data"]
                })
                yield history
                #yield f"Agent answer: {rsp["data"]}"

        else:
            history.append({
                "role": MessageRole.AGENT,
                "Agent profile": 'data_manager',
                "content": "No valid response received."
            })
            yield history
            #yield "No valid response received.Try again!"

    except Exception as e:
        traceback.print_exc()
        history.append({
            "role": MessageRole.AGENT,
            "Agent profile": 'error',
            "content": f"An error occurred: {str(e)}"
        })
        yield history
        #yield f"An error occurred: {str(e)}"


demo = gr.ChatInterface(fn=chat_demo, title="Testing Network of Agents with AutoGen", type="messages", autofocus=False)
if __name__ == "__main__":
    demo.launch()
