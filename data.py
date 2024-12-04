import warnings
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_openai import ChatOpenAI
import json
from pprint import pprint
from keys import openai_api_key, serper_api_key
warnings.filterwarnings('ignore')


# Set environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPER_API_KEY"] = serper_api_key

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

'''# Agents
web_agent = Agent(
    role="Internship Opportunity Finder",
    goal=(
        "Identify {num_leads} internship or associate opportunities in positions {positions} "
        "with qualifications {qualifications} and skills {skills}, located in {location}. "
        "Exclude positions related to {excluded_positions}."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Specializes in researching and identifying internship or associate opportunities for master's level candidates "
        "in specified fields and locations. Focuses on finding relevant job postings and company websites."
    )
)'''

# Agents with industry
web_agent = Agent(
    role="Internship Opportunity Finder",
    goal=(
        "Identify {num_leads} internship or associate opportunities in positions {positions} within industries {industries} "
        "with qualifications {qualifications} and skills {skills}, located in {location}. "
        "Exclude positions related to {excluded_positions}."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Specializes in researching and identifying internship or associate opportunities for master's level candidates "
        "in specified fields, industries, and locations. Focuses on finding relevant job postings and company websites."
    )
)


info_extractor = Agent(
    role="Information Extractor",
    goal="Extract company websites, roles, positions, and deadlines (if available) from job postings and company websites.",
    tools=[scrape_tool],
    verbose=True,
    backstory=(
        "Expert in analyzing job postings and company websites to locate and extract key information needed for applications."
    )
)

data_validator = Agent(
    role="Data Validation Specialist",
    goal=(
        "Validate and organize extracted information. "
        "Ensure all data is accurate, complete, and outputted as a JSON array of objects with keys: "
        "'Company Website', 'Role', 'Position Title', and 'Deadline'."
    ),
    tools=[],
    verbose=True,
    backstory=(
        "With an analytical mindset and a focus on accuracy, you specialize in validating and structuring data."
    )
)

# Tasks
website_task = Task(
    description=(
        "Search for {num_leads} internship or associate opportunities in positions {positions} "
        "with qualifications {qualifications} and skills {skills}, located in {location}. "
        "Exclude positions related to {excluded_positions}."
    ),
    expected_output="A list of internship or associate job postings or company websites relevant to the specified criteria.",
    human_input=True,
    output_file="opportunities.json",
    agent=web_agent,
)

info_extraction_task = Task(
    description="Extract company websites, roles, positions, and deadlines (if available) from the job postings and company websites identified.",
    expected_output="Details including company website, role, position, and application deadline.",
    human_input=False,
    output_file="extracted_info.json",
    agent=info_extractor,
)

validate_task = Task(
    description=(
        "Validate the extracted information. Ensure all data is accurate, complete, and formatted as a JSON array "
        "with keys: 'Company Website', 'Role', 'Position Title', and 'Deadline'."
    ),
    expected_output="A JSON array of validated and organized information ready for output.",
    human_input=False,
    output_file="validated_info.json",
    agent=data_validator,
)

# Inputs
positions = ["Quant Researcher", "Data Science", "Data Analyst", "Consulting"]
conditions = ["internship", "associate"]
qualifications = ["Masters"]
skills = ["Python", "Math", "SQL", "Matlab", "Tableau", "PowerBi"]
excluded_positions = ["Marketing", "Sales", "HR"]

location = input("Enter the location (e.g., New York City): ")
num_leads = int(input("Enter the number of leads to generate: "))
# Add industries input
industries_input = input("Enter the industries (e.g., Insurance, Hedge Funds, Banks): ")
industries = [industry.strip() for industry in industries_input.split(',')]

inputs = {
    "positions": positions,
    "conditions": conditions,
    "qualifications": qualifications,
    "skills": skills,
    "industries": industries,
    "location": location,
    "num_leads": num_leads,
    "excluded_positions": excluded_positions,
}
# Crew
contact_extraction_crew = Crew(
    agents=[web_agent, info_extractor, data_validator],
    tasks=[website_task, info_extraction_task, validate_task],
    manager_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key),
    process=Process.hierarchical,
    verbose=True,
)

# Execute
result = contact_extraction_crew.kickoff(inputs=inputs)

# Check for output file existence
try:
    with open("validated_info.json") as f:
        validated_info = json.load(f)
    pprint(validated_info)
    # Optionally, save to CSV
    import csv
    keys = validated_info[0].keys()
    with open('validated_info.csv', 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(validated_info)
except FileNotFoundError:
    print("No validated information found.")
except json.JSONDecodeError as e:
    print(f"Error loading JSON data: {e}")
    with open("validated_info.json") as f:
        content = f.read()
    print("Content of validated_info.json:")
    print(content)
