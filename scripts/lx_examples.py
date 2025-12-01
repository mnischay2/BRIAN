import langextract as lx

Ex = [
        # --- ACADEMIC / TEXTBOOK ---
        lx.data.ExampleData(
            text="In 1915, Einstein published General Relativity. It explains gravity as curvature of spacetime.",
            extractions=[
                lx.data.Extraction(extraction_class="Topic", extraction_text="General Relativity"),
                lx.data.Extraction(extraction_class="Person", extraction_text="Einstein"),
                lx.data.Extraction(extraction_class="Date", extraction_text="1915"),
                lx.data.Extraction(extraction_class="Definition", extraction_text="gravity as curvature of spacetime"),
            ]
        ),
        lx.data.ExampleData(
            text="Photosynthesis occurs in the chloroplasts. It converts light energy into chemical energy.",
            extractions=[
                lx.data.Extraction(extraction_class="Process", extraction_text="Photosynthesis"),
                lx.data.Extraction(extraction_class="Location", extraction_text="chloroplasts"),
                lx.data.Extraction(extraction_class="Function", extraction_text="converts light energy into chemical energy"),
            ]
        ),
        lx.data.ExampleData(
            text="The Treaty of Versailles was signed on June 28, 1919, effectively ending World War I.",
            extractions=[
                lx.data.Extraction(extraction_class="Event", extraction_text="Treaty of Versailles"),
                lx.data.Extraction(extraction_class="Date", extraction_text="June 28, 1919"),
                lx.data.Extraction(extraction_class="Significance", extraction_text="ending World War I"),
            ]
        ),
        lx.data.ExampleData(
            text="Newton's Second Law states that Force equals mass times acceleration (F=ma).",
            extractions=[
                lx.data.Extraction(extraction_class="Law", extraction_text="Newton's Second Law"),
                lx.data.Extraction(extraction_class="Formula", extraction_text="F=ma"),
                lx.data.Extraction(extraction_class="Definition", extraction_text="Force equals mass times acceleration"),
            ]
        ),

        # --- RESUME / CV ---
        lx.data.ExampleData(
            text="Proficient in Python, Java, and Docker. Certified AWS Solutions Architect.",
            extractions=[
                lx.data.Extraction(extraction_class="Skill", extraction_text="Python"),
                lx.data.Extraction(extraction_class="Skill", extraction_text="Java"),
                lx.data.Extraction(extraction_class="Skill", extraction_text="Docker"),
                lx.data.Extraction(extraction_class="Certification", extraction_text="AWS Solutions Architect"),
            ]
        ),
        lx.data.ExampleData(
            text="Jane Doe. Senior Software Engineer at Google from 2018 to Present.",
            extractions=[
                lx.data.Extraction(extraction_class="Person", extraction_text="Jane Doe"),
                lx.data.Extraction(extraction_class="Role", extraction_text="Senior Software Engineer"),
                lx.data.Extraction(extraction_class="Company", extraction_text="Google"),
                lx.data.Extraction(extraction_class="DateRange", extraction_text="2018 to Present"),
            ]
        ),
        lx.data.ExampleData(
            text="B.S. in Computer Science from MIT, Graduated 2015. GPA 3.9.",
            extractions=[
                lx.data.Extraction(extraction_class="Degree", extraction_text="B.S. in Computer Science"),
                lx.data.Extraction(extraction_class="University", extraction_text="MIT"),
                lx.data.Extraction(extraction_class="GraduationYear", extraction_text="2015"),
                lx.data.Extraction(extraction_class="Metric", extraction_text="GPA 3.9"),
            ]
        ),
        lx.data.ExampleData(
            text="Implemented a CI/CD pipeline using Jenkins that reduced deployment time by 40%.",
            extractions=[
                lx.data.Extraction(extraction_class="Project", extraction_text="CI/CD pipeline"),
                lx.data.Extraction(extraction_class="Tool", extraction_text="Jenkins"),
                lx.data.Extraction(extraction_class="Achievement", extraction_text="reduced deployment time by 40%"),
            ]
        ),

        # --- STORYBOOK / NARRATIVE ---
        lx.data.ExampleData(
            text="Sherlock Holmes stood in the center of the Baker Street apartment, holding a magnifying glass.",
            extractions=[
                lx.data.Extraction(extraction_class="Character", extraction_text="Sherlock Holmes"),
                lx.data.Extraction(extraction_class="Location", extraction_text="Baker Street apartment"),
                lx.data.Extraction(extraction_class="Object", extraction_text="magnifying glass"),
            ]
        ),
        lx.data.ExampleData(
            text="As the dragon Smaug awoke, Bilbo Baggins trembled in the shadows of the Lonely Mountain.",
            extractions=[
                lx.data.Extraction(extraction_class="Character", extraction_text="Smaug"),
                lx.data.Extraction(extraction_class="Character", extraction_text="Bilbo Baggins"),
                lx.data.Extraction(extraction_class="Location", extraction_text="Lonely Mountain"),
                lx.data.Extraction(extraction_class="Action", extraction_text="awoke"),
            ]
        ),
        lx.data.ExampleData(
            text="Eliza realized the map was a fake. She threw it into the fireplace.",
            extractions=[
                lx.data.Extraction(extraction_class="Character", extraction_text="Eliza"),
                lx.data.Extraction(extraction_class="PlotPoint", extraction_text="map was a fake"),
                lx.data.Extraction(extraction_class="Action", extraction_text="threw it into the fireplace"),
            ]
        ),
        lx.data.ExampleData(
            text="The spaceship landed on Mars. The red dust swirled around the landing gear.",
            extractions=[
                lx.data.Extraction(extraction_class="Vehicle", extraction_text="spaceship"),
                lx.data.Extraction(extraction_class="Location", extraction_text="Mars"),
                lx.data.Extraction(extraction_class="SettingDetail", extraction_text="red dust swirled"),
            ]
        ),

        # --- RESEARCH PAPER / TECHNICAL ---
        lx.data.ExampleData(
            text="We utilized a Transformer architecture with 12 layers and 768 hidden units.",
            extractions=[
                lx.data.Extraction(extraction_class="Methodology", extraction_text="Transformer architecture"),
                lx.data.Extraction(extraction_class="Parameter", extraction_text="12 layers"),
                lx.data.Extraction(extraction_class="Parameter", extraction_text="768 hidden units"),
            ]
        ),
        lx.data.ExampleData(
            text="The accuracy on the test set reached 98.5%, outperforming the baseline by 2%.",
            extractions=[
                lx.data.Extraction(extraction_class="Metric", extraction_text="accuracy"),
                lx.data.Extraction(extraction_class="Value", extraction_text="98.5%"),
                lx.data.Extraction(extraction_class="Result", extraction_text="outperforming baseline by 2%"),
            ]
        ),
        lx.data.ExampleData(
            text="Figure 3 illustrates the correlation between temperature and enzyme activity.",
            extractions=[
                lx.data.Extraction(extraction_class="Figure", extraction_text="Figure 3"),
                lx.data.Extraction(extraction_class="Relationship", extraction_text="correlation between temperature and enzyme activity"),
            ]
        ),
        lx.data.ExampleData(
            text="Samples were collected from the Pacific Ocean at a depth of 2000 meters.",
            extractions=[
                lx.data.Extraction(extraction_class="SampleSource", extraction_text="Pacific Ocean"),
                lx.data.Extraction(extraction_class="Condition", extraction_text="depth of 2000 meters"),
            ]
        ),

        # --- FINANCIAL / BUSINESS / LEGAL ---
        lx.data.ExampleData(
            text="Revenue for Q3 2023 was $15.4 million, an increase of 12% YoY.",
            extractions=[
                lx.data.Extraction(extraction_class="Metric", extraction_text="Revenue"),
                lx.data.Extraction(extraction_class="Period", extraction_text="Q3 2023"),
                lx.data.Extraction(extraction_class="Amount", extraction_text="$15.4 million"),
                lx.data.Extraction(extraction_class="Trend", extraction_text="increase of 12% YoY"),
            ]
        ),
        lx.data.ExampleData(
            text="This Agreement is made between Acme Corp ('Seller') and Beta Ltd ('Buyer').",
            extractions=[
                lx.data.Extraction(extraction_class="DocumentType", extraction_text="Agreement"),
                lx.data.Extraction(extraction_class="Party", extraction_text="Acme Corp"),
                lx.data.Extraction(extraction_class="Role", extraction_text="Seller"),
                lx.data.Extraction(extraction_class="Party", extraction_text="Beta Ltd"),
                lx.data.Extraction(extraction_class="Role", extraction_text="Buyer"),
            ]
        ),
        lx.data.ExampleData(
            text="The project roadmap includes three phases: Discovery, Development, and Launch.",
            extractions=[
                lx.data.Extraction(extraction_class="Topic", extraction_text="project roadmap"),
                lx.data.Extraction(extraction_class="Phase", extraction_text="Discovery"),
                lx.data.Extraction(extraction_class="Phase", extraction_text="Development"),
                lx.data.Extraction(extraction_class="Phase", extraction_text="Launch"),
            ]
        ),
        lx.data.ExampleData(
            text="EBITDA margin improved to 25% due to cost-cutting measures in logistics.",
            extractions=[
                lx.data.Extraction(extraction_class="Metric", extraction_text="EBITDA margin"),
                lx.data.Extraction(extraction_class="Value", extraction_text="25%"),
                lx.data.Extraction(extraction_class="Cause", extraction_text="cost-cutting measures in logistics"),
            ]
        ),

        # --- MEDICAL / HEALTH ---
        lx.data.ExampleData(
            text="Patient presented with acute bronchitis and was prescribed Amoxicillin 500mg.",
            extractions=[
                lx.data.Extraction(extraction_class="Condition", extraction_text="acute bronchitis"),
                lx.data.Extraction(extraction_class="Medication", extraction_text="Amoxicillin"),
                lx.data.Extraction(extraction_class="Dosage", extraction_text="500mg"),
            ]
        ),
        lx.data.ExampleData(
            text="The MRI scan revealed a minor fracture in the tibia.",
            extractions=[
                lx.data.Extraction(extraction_class="Procedure", extraction_text="MRI scan"),
                lx.data.Extraction(extraction_class="Finding", extraction_text="minor fracture"),
                lx.data.Extraction(extraction_class="Anatomy", extraction_text="tibia"),
            ]
        ),
         lx.data.ExampleData(
            text="Symptoms include fever, dry cough, and fatigue appearing 2-14 days after exposure.",
            extractions=[
                lx.data.Extraction(extraction_class="Symptom", extraction_text="fever"),
                lx.data.Extraction(extraction_class="Symptom", extraction_text="dry cough"),
                lx.data.Extraction(extraction_class="Symptom", extraction_text="fatigue"),
                lx.data.Extraction(extraction_class="Timeframe", extraction_text="2-14 days after exposure"),
            ]
        ),
    ]
    