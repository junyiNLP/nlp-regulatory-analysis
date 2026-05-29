import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent  # points to BERT_training
MODEL_DIR = BASE_DIR / "stage1_training_5_single_label" / "model_training_5_single_label"


# ============================================================
# LOAD MODEL + TOKENIZER
# ============================================================

print("🔹 Loading trained semantic model...")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.eval()

# Load label mapping
config_path = MODEL_DIR / "config.json"
with open(config_path, "r") as f:
    config_data = json.load(f)

id2label = {int(k): v for k, v in config_data["id2label"].items()}

# ============================================================
# DEVICE SELECTION
# ============================================================

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"🔹 Using device: {device}")
model.to(device)

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(text):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,   # MATCH TRAINING
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

    return id2label[pred_id], confidence


# ============================================================
# RUN PREDICTION
# ============================================================

# ============================================================
# 🟦 Test the model on ECA dataset
# ============================================================

#4_6
#TEXT = "Copyright is not adequately enforced online in the Republic of Congo. It is reported that there are high levels of piracy since the 1990s, with complaints also on the fact that the Republic of Congo government computers use unlicensed software."

#TEXT = "The Republic of Congo has not signed the World Intellectual Property Organization (WIPO) Copyright Treaty."

#TEXT = "The Republic of Congo has not signed the World Intellectual Property Organization (WIPO) Performances and Phonograms Treaty."

#TEXT = "According to the Electronic Communication Law (Art. 6), the electronic communications activities are exercised freely in Congo. There are no discriminatory conditions for foreign companies. However, the supply of electronic communications networks and services is conditional on obtaining, depending on the case of: licence, authorization, agreement, declaration, experimentation. It is reported that in the Republic of Congo different minimum capital requirements are established to obtain different types of licenses:- 10.000.000.000 F.CFA for 3G and 4G (approx. USD 15,196,000);- 11.000.000.000 F.CFA for 2G (approx. USD 16,716,000)."

#TEXT = "The Congo does not impose any import ban on ICT goods. As a Partner State of the EAC Customs Union, the Congo is legally bound by the East African Community Customs Management Act, 2004 (EACCMA) and the East African Community Common External Tariff (EAC CET). Therefore, it applies the EAC import regime on ICT goods and digital services and abides by the regional Customs Management, and Customs administration."

#Cameroon 2_3
#=TEXT = "Sub-section VII (of the National Preference) of the New Public Procurement Code of Decree No. 2018/366 imposes restrictions for participation of foreign companies in public procurement. Art. 106 states that: When awarding a contract within the framework of an international consultation, a margin of preference is granted, to equivalent offers and in order of priority to the tenders presented by: a) a natural person of Cameroonian nationality or a legal person under Cameroonian law; b) a company whose capital is wholly or mainly held by persons of Cameroonian nationality; c) a natural person or a legal person justifying an economic activity on the territory of Cameroon; d) a group of companies associating Cameroonian companies; Tenders are considered equivalent when they have fulfilled the required technical conditions; For quantifiable works and services contracts, the national preference margin is 10% for the companies referred to in paragraph 1 above; For supply contracts, the national preference criterion can only be taken into account if the supply undergoes a transformation at the local or regional level of at least 15%;There is no national preference for non-quantifiable service contracts including intellectual services;National preference can only be applied when the tender documents provide for it."


#Cameroon 4_5
#TEXT = "Cameroon has a clear regime of copyright exceptions, which enable the lawful use of copyrighted work by others without obtaining permission (Art. 29 of Law No. 2000/011). The regime follows the fair use model. In addition, the Revised Bangui Agreement, which is a regional intellectual property law that is not only a regional convention applicable in all member states but also serves as a national intellectual property law in Cameroon and each of the method_combine_dataset member states, contains provisions on copyright in Annex VII: Chapter IV (Limitations to Economic Rights) and establishes a regime of copyright exceptions."


#DRC 5_5 / 9_1
#TEXT = "With the Law No. 013/2002 of 2002, which governs the telecommunications sector, the government is given the power to take over the means of communication in the interest of national security. Art. 46 provides that the State may prohibit the use of telecommunication facilities, in full or in part, for any period of time, as it deems fit, in the interests of public security or national defence, the public telecommunications service, or for any method_combine_dataset reason. "

#TEXT = "In June 2018, the Minister of Posts, Telecommunications and Information, signed a Ministerial Order strengthening the control of online media (Ministerial Order No 011/CAB/M-CM/LOM/2018)."

#Algeria_2_3
#TEXT = "According to Art. 5 of Presidential Decree No. 15-247, in order to ensure the efficiency of the order and the proper use of public funds, public contracts must respect the principles of free access to public procurement, equal treatment of candidates regardless of nationality, and transparency of procedures. However, Art. 85 of the Presidential Decree also requires that the governmental purchaser, in the process of drafting the eligibility criteria and the offer evaluation procedure, must consider whether national production and/or national entities can meet the needs of the contract – and therefore by that extent also favour batches or products that can be subcontracted or acquired on the local Algerian market. In circumstances when relying on national production and/or entities is not possible, foreign companies can bid with certain limitations. Imported products may only be used in circumstances when the equivalent of the products is either not available at a national level or the national products do not possess the acceptable quality standards."

#pillar4_Intellectual Property Rights
#TEXT = "Under the Patent Rules No. S19/2012 and Patents Order No. S57/2011, Brunei Darussalam applies a transparent, non-discriminatory patent-filing process through authorized local agents, while residents wishing to seek patents abroad must obtain prior approval from the Brunei Registry of Patents and wait two months after filing locally."

#TEXT = "Robust civil and administrative procedures, remedies, and provisional measures are readily available for patent and copyright enforcement, including injunctions, damages, destruction of infringing goods, and seizure of infringing copies."

#TEXT = "Brunei Darussalam is a contracting state to the Patent Cooperation Treaty, facilitating streamlined international patent filings. Based on fair dealing provisions in Sections 33 and 34 of the Emergency (Copyright) Order No.S14/2000 and Copyright (Amendment) Order No.S92/2013, clear copyright exceptions exist for research, private study, criticism, reviews, and news reporting."

#4_8 and
#TEXT = "Brunei Darussalam is a signatory to the WIPO Copyright Treaty and has acceded to the WIPO Performances and Phonograms Treaty. "

#pillar_9_content_access
#TEXT = "Government authorities in Brunei Darussalam routinely block or filter commercial content deemed to violate national values, public order, or public harmony from websites under Section 11 of the Broadcasting (Class Licence) Notification 2001 and Sections 1 and 6 of the Broadcasting Services Programmes Code."

#Niue
#TEXT = "There is no ceiling on the maximum amount that can be paid by electronic payment methods."

#Algeria 4_4
#TEXT = ''' The Copyright and Neighbouring Rights Law does not provide any fair use/fair dealing model for copyright exceptions. However, Art.33-53 provide a list of wide exceptions. These include:  materials that are printed, audio, audio-visual or any method_combine_dataset form prepared for school or university education;  work converted for personal or family purposes;  usage of decorative or illustrative drawing of a literary or artistic work in a publication, in an audio or audio video recording or in audio or audio video programs meant for teaching or professional training as long as it is intended to achieve the targeted purpose; free acting or performing of a work; libraries and document keeping centers, not aiming at making direct or indirect commercial profits, can reproduce a work without the author’s or right owner’s permission in response to a request from another library or document keeping center, or to maintain the work’s copy, or to compensate a damaged, lost or void one;  and mass media institutions reproducing, circulating or conveying to the public articles on daily events published by newspapers, radio or television, without permission from the author or reward to him, provided that the author and the source are indicated, unless there is an express provision prohibiting its use for such purposes.  '''

# #Algeria 7_5
# TEXT = '''  According to Art. 3 of Law No. 09-04, subject to the legal provisions guaranteeing the secrecy of correspondence and communications, implementation of technical devices carrying out operations of surveillance of electronic communications, collection and recording in real time of their content as well as searches and seizures in a computer system may be carried out for purposes of protection of public order, for investigations or after judicial request. Art. 4 lists instances when such surveillance may be carried out, including:
# - to prevent offenses qualified as terrorist or subversive acts and offenses against State security;
# - when there is information on a probable attack on a computer system representing a threat to public order, national defence, State institutions or the national economy;
# - for the purposes of investigations and judicial information when it is difficult to arrive at results relevant to the research in progress without resorting to electronic surveillance;
# - in connection with the execution of requests for international legal assistance. '''

# #Algeria 8_3
# TEXT = ''' Art. 11 of Law No. 09-04 requires ICT service providers to store data allowing the identification of users of their services for a period of one year after the registration. This requirement covers all service providers, defined in Art. 2 to include any public or private entity which offers users of its services the possibility of communicating by means of a computer system and/or a telecommunications system and any method_combine_dataset entity processing or storing computer data for communication services or their users.  '''

#Algeria 9_1
# TEXT = ''' It is reported that Algeria has a history of its government blocking sites for various reasons, including the prevention of exam cheating. Moreover, web-based Algerian media outlets have been required to be based within the country and they have been asked to inform authorities of any "illegal content". Some of the websites that have been blocked in the past include YouTube, Facebook, and Twitter, as well as news websites such as France 24 and Al-Jazeera.  '''

#Malaysia 10_2
# TEXT = '''- Import of machines and parts for optical disc mastering and replicating under Subheading 8479.83.00, 8479.89.69, 8485.30.90, 8485.80.00, and 8479.90. Import approval required from the Ministry of Domestic Trade and Cost of Living. Control to prevent unauthorized production of optical media.
# Digital and Networking Devices:
# - Security Devices such as Firewalls, intrusion detection/prevention systems under Subheading 8471.49.90, 8471.50.90. Import must adhere to specific conditions.
# - RFID Devices: Includes readers and tags under Subheading 8471.60.90, 8471.90.90. Subject to inspection and approval.
# - Wireless Signal Amplifiers: Cellular/wireless boosters or repeaters under Subheading 8525.60.00, 8517.62.49. Importation restrictions apply based on product specification.
# - Television and Display Equipment: Items Used televisions and video display units with TV tuners under Subheading 8528.72.91, 8528.72.92, 8528.72.99, 8528.73.00. Must comply with specific approval letters.
# - Internet Protocol Set-Top Boxes (STBs) in Subheading: 8528.71.11. Requires conformity to import control laws and regulations '''


#Malaysia Act 588 section 233
TEXT = ''' (1) A person who—
(a) by means of any network facilities or network service or
applications service knowingly—
(i) makes, creates or solicits; and
(ii) initiates the transmission of,
any comment, request, suggestion or method_combine_dataset communication
which is obscene, indecent, false, menacing or offensive
in character with intent to annoy, abuse, threaten or
harass another person; or
Communications and Multimedia
119
(b) initiates a communication using any applications service,
whether continuously, repeatedly or otherwise, during
which communication may or may not ensue, with or
without disclosing his identity and with intent to annoy,
abuse, threaten or harass any person at any number or
electronic address,
commits an offence.
(2) A person who knowingly—
(a) by means of a network service or applications service
provides any obscene communication for commercial
purposes to any person; or
(b) permits a network service or applications service under
the person’s control to be used for an activity described
in paragraph (a),
commits an offence.
(3) A person who commits an offence under this section shall,
on conviction, be liable to a fine not exceeding fifty thousand
ringgit or to imprisonment for a term not exceeding one year or
to both and shall also be liable to a further fine of one thousand
ringgit for every day during which the offence is continued after
conviction.'''


label, conf = predict(TEXT)

print("\n=======================")
print("🔮 Prediction Result")
print("=======================\n")
print(f"Input text: {TEXT}")
print(f"Predicted label: {label}")
print(f"Confidence: {conf:.4f}")
print("\n=======================")
