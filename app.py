from flask import Flask, render_template, request
import pickle
import re
import pdfplumber
import cv2
import pytesseract as pyt
import numpy as np


app = Flask(__name__)

pyt.pytesseract.tesseract_cmd = r'C:/Users/IBE/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'


# Load models
try:
    rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
    tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
    rf_classifier_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
    tfidf_vectorizer_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    raise


def pdf_to_text(file_path):
    try:
        # Open the PDF file
        with pdfplumber.open(file_path) as pdf:
            extracted_text = ""
            # Iterate through all pages in the PDF
            for page in pdf.pages:
                # Extract text from each page
                extracted_text += page.extract_text() + "\n"
            return extracted_text
    except Exception as e:
        return f"An error occurred while extracting text: {e}"
    
def image_to_text(file):
    # Read the file as a numpy array
    file_bytes = np.frombuffer(file.read(), np.uint8)
    
    # Decode the numpy array into an image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode the image. Please upload a valid JPG file.")
    
    # Now process the image (example with pytesseract)
    text = pyt.image_to_string(img)
    return text


def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', '', cleanText)
    cleanText = re.sub('@\S+', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', r' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    print("Cleaned Resume Text:", resume_text)  # Debugging line
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category

def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_recommendation.transform([resume_text])
    predicted_job = rf_classifier_recommendation.predict(resume_tfidf)[0]
    return predicted_job

def extract_name_from_resume(text):
    name = None
    pattern = r"(\b[A-Z]\.\s?[A-Z][a-z]+\b|\b[A-Z][a-z]+\s[A-Z]\b|\b[A-Z]\s[A-Z][a-z]+\b|\b[A-Z][a-z]+\s[A-Z]\.\b|\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"
    match = re.search(pattern, text)
    if match:
        name = match.group()
    return name

def extract_contact_number_from_resume(text):
    contact_number = None
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        contact_number = match.group()
    return "+" + contact_number

def extract_email_from_resume(text):
  email = None

  pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
  match = re.search(pattern, text)
  if match:
    email = match.group()

  return email


def extract_skills_from_resume(text):
  skills_list = [
    # Programming Languages
    "Python","MySQL", "JavaScript", "Java", "C", "C++", "Ruby", "PHP", "Go", "Swift", "Kotlin", "R", "Rust", "TypeScript", "SQL", "C#", "Scala",
    # Web Development
    "HTML", "CSS", "JavaScript", "React", "Angular", "Vue.js", "Node.js", "Express.js", "Django", "Flask", "Ruby on Rails", "ASP.NET", "Bootstrap",
    # Frontend Development
    "HTML5", "CSS3", "SASS", "LESS", "JQuery", "AJAX", "WebSockets", "WebRTC", "Responsive Web Design", "Progressive Web Apps (PWA)",
    # Backend Development
    "Node.js", "PHP", "Ruby", "Java", "C#", "Python", "Go", "Spring Boot", "ASP.NET Core", "NestJS", "Express.js", "GraphQL", "RESTful APIs",
    # Database Management
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "Redis", "Cassandra", "MariaDB", "NoSQL", "Graph Databases", "DynamoDB",
    # Cloud Computing
    "AWS", "Azure", "Google Cloud Platform", "Heroku", "IBM Cloud", "Oracle Cloud", "CloudFormation", "Kubernetes", "Docker", "CI/CD", "Terraform", "OpenStack",
    # DevOps Tools
    "Jenkins", "Ansible", "Chef", "Puppet", "Kubernetes", "Docker", "Git", "GitHub", "GitLab", "CircleCI", "Travis CI", "Maven", "Gradle",
    # Operating Systems
    "Linux", "Unix", "Windows", "macOS", "Android", "iOS", "CentOS", "Ubuntu", "Debian", "Red Hat", "Fedora", "Docker",
    # Networking
    "TCP/IP", "DNS", "HTTP/HTTPS", "FTP", "SSH", "VPN", "LAN/WAN", "Load Balancing", "Network Security", "Firewalls", "Routing", "Switching", "IP Addressing",
    # Cybersecurity
    "Cryptography", "Network Security", "Penetration Testing", "Firewalls", "Intrusion Detection Systems (IDS)", "Intrusion Prevention Systems (IPS)", "Risk Management",
    "Identity and Access Management (IAM)", "Security Information and Event Management (SIEM)", "Ethical Hacking", "Malware Analysis", "Phishing Protection",
    # Mobile Development
    "iOS Development", "Android Development", "Swift", "Kotlin", "Flutter", "React Native", "Xamarin", "Mobile UI/UX", "Android Studio", "Xcode",
    # Data Science & Machine Learning
    "Machine Learning", "Deep Learning", "Data Mining", "Natural Language Processing", "TensorFlow", "PyTorch", "Scikit-learn", "Keras", "Pandas", "NumPy",
    "Matplotlib", "Seaborn", "Jupyter Notebooks", "Data Cleaning", "Model Deployment", "Big Data", "Spark", "Hadoop", "AI", "Data Visualization",
    # Artificial Intelligence
    "AI", "Neural Networks", "Reinforcement Learning", "Computer Vision", "Speech Recognition", "Chatbots", "Robotics", "Natural Language Processing (NLP)",
    # Blockchain & Cryptocurrency
    "Blockchain", "Smart Contracts", "Ethereum", "Solidity", "Bitcoin", "Cryptocurrency", "Decentralized Applications (dApps)", "IPFS", "DeFi", "NFTs",  
    # Game Development
    "Unity", "Unreal Engine", "C#", "C++", "Game Design", "Game Engines", "3D Modeling", "AI in Games", "Multiplayer Game Development", "Augmented Reality (AR)", "Virtual Reality (VR)",
    # Business Intelligence (BI)
    "Tableau", "Power BI", "Qlik", "Looker", "Data Warehousing", "ETL", "Data Pipelines", "Business Analytics", "Dashboards", "Data Reporting", "Excel", "PowerPivot",
    # UI/UX Design
    "UI Design", "UX Design", "Wireframing", "Prototyping", "Adobe XD", "Sketch", "Figma", "InVision", "User Research", "A/B Testing", "User Flows",
    # Software Testing
    "Test Automation", "Manual Testing", "Selenium", "JUnit", "TestNG", "Mocha", "Jest", "Cypress", "Appium", "Load Testing", "Performance Testing", "Unit Testing", "API Testing",
    # Agile & Project Management
    "Scrum", "Kanban", "Agile", "JIRA", "Trello", "Confluence", "Lean", "Project Management", "Sprint Planning", "Continuous Integration", "Continuous Delivery",
    # IT Support & Helpdesk
    "Technical Support", "IT Service Management (ITSM)", "Ticketing Systems", "Network Troubleshooting", "Hardware Repair", "Software Installation", "System Administration", "Remote Support",
    # Virtualization
    "VMware", "Hyper-V", "VirtualBox", "Docker", "VMs", "Containers", "Cloud Computing Virtualization", "VM Clustering", "KVM", "Xen",
    # Internet of Things (IoT)
    "IoT", "Sensor Networks", "Embedded Systems", "Arduino", "Raspberry Pi", "Home Automation", "Edge Computing", "Cloud IoT", "Wireless Technologies",
    # IT Management & Leadership
    "IT Strategy", "IT Governance", "ITIL", "Change Management", "Service Management", "Digital Transformation", "IT Budgeting", "Vendor Management", "Team Leadership",
    # Robotics Process Automation (RPA)
    "Automation Anywhere", "UiPath", "Blue Prism", "WorkFusion", "Robotics Process Automation", "AI Automation", "Bots Development", "RPA Deployment",
    # Data Engineering
    "ETL", "Data Warehousing", "Big Data", "Apache Kafka", "Apache Flume", "Apache Spark", "Hadoop", "Data Pipelines", "Apache Airflow", "Data Modeling", "Data Lakes",
    # Hardware & Embedded Systems
    "Microcontrollers", "Arduino", "Raspberry Pi", "FPGA", "Circuit Design", "Embedded C", "RTOS", "Device Firmware", "IoT Devices",  
    # Cloud Security
    "Cloud Security", "IAM", "Cloud Architecture", "Security Audits", "Incident Response", "Security Best Practices", "Encryption", "Cloud Firewalls",
    # IT Compliance & Regulatory
    "GDPR", "HIPAA", "PCI-DSS", "SOX", "ISO 27001", "NIST", "COBIT", "IT Risk Management", "SOC 2 Compliance", "Data Privacy",
    ]
  
  skills = set()

  for skill in skills_list:
    pattern = r"\b{}\b".format(re.escape(skill))
    match = re.search(pattern,text,re.IGNORECASE)
    if match:
      skills.add(match.group())
  return list(skills)


def extract_education_from_resume(text):
    education = set()  # Use a set to avoid duplicates

    # Define Degrees and Certifications and Fields of Study
    degrees_and_certifications = [
        "Bachelor of Science", "B.Sc.", "Bachelor of Arts", "B.A.", "Master of Science", "M.Sc.",
        "Master of Arts", "M.A.", "Master of Business Administration", "MBA", "Doctor of Philosophy",
        "Ph.D.", "Associate Degree", "Diploma", "Certificate", "Professional Certification", "BSc","BE", 
        "B.E.", "B.Tech", "M.Sc", "M.E.", "M.Tech", "Ph.D", "PhD"
    ]

    fields_of_study = [
        "Computer Science", "Information Technology", "Software Engineering", "Electrical Engineering",
        "Mechanical Engineering", "Civil Engineering", "Business Administration", "Marketing",
        "Finance", "Accounting", "Economics", "Psychology", "Biology", "Physics", "Mathematics",
        "Statistics", "Chemistry", "Environmental Science", "Nursing", "Medicine",
        "Pharmacy", "Law", "Architecture", "Data Science", "Artificial Intelligence", "Machine Learning",
        "Cybersecurity", "Web Development", "Game Development", "Graphic Design", "Digital Marketing"
    ]
    
    # Create patterns for degrees and fields of study
    degree_pattern = r"(?i)\b(?:{})\b".format("|".join(map(re.escape, degrees_and_certifications)))
    field_pattern = r"(?i)\b(?:{})\b".format("|".join(map(re.escape, fields_of_study)))

    # Find all matches for degrees and fields of study in the text
    degree_matches = re.finditer(degree_pattern, text)
    field_matches = re.finditer(field_pattern, text)
    
    # Extract all positions (indices) of the matches
    degree_positions = [match.span() for match in degree_matches]
    field_positions = [match.span() for match in field_matches]

    # Now, check if degrees and fields of study are near each other
    for degree_start, degree_end in degree_positions:
        for field_start, field_end in field_positions:
            # Check if the matches are within a range of characters (e.g., 50 characters)
            if abs(degree_start - field_start) <= 50:
                # If they are near each other, add both degree and field to the result
                education.add(text[degree_start:degree_end].strip())
                education.add(text[field_start:field_end].strip())

    return list(education)


@app.route('/')
def resume():
    return render_template('resume.html')

@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename

        if filename.endswith('.pdf'):
            text = pdf_to_text(file)
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
            text = image_to_text(file)
        else:
            return render_template('resume.html', message='Invalid file format. Please upload a PDF, TXT, or JPG file.')

        predicted_category = predict_category(text)
        recommended_job = job_recommendation(text)
        name = extract_name_from_resume(text)
        phone = extract_contact_number_from_resume(text)
        email = extract_email_from_resume(text)
        
        skills = extract_skills_from_resume(text)
        education = extract_education_from_resume(text)
        
        return render_template('resume.html', predicted_category=predicted_category, recommended_job=recommended_job,
                               phone=phone, email=email, name=name, skills=skills, education=education)
    else:
        return render_template('resume.html', message='No Resume uploaded.')


if __name__ == '__main__':
    app.run(debug=True)
