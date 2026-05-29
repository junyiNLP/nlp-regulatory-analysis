
# we can test the model performance on single paragraph of text

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent  # points to BERT_training
MODEL_DIR = BASE_DIR / "stage1_training_3_single_label" / "model_training_3"

# ============================================================
# LOAD MODEL + TOKENIZER
# ============================================================

print("🔹 Loading Stage-1 Semantic model...")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.eval()

# Load label mapping
config_path = MODEL_DIR / "config.json"
with open(config_path, "r") as f:
    config_data = json.load(f)

id2label = {int(k): v for k, v in config_data["id2label"].items()}

# Device selection
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"🔹 Using device: {device}")
model.to(device)

# ============================================================
# 🟦 Test the model on random pick-up text
# ============================================================

TEXT = "Online advertisements in the country are subject to strict regulations, including prohibitions on alcohol, illegal drugs, non-Islamic religions, immoral or anti-social behaviour, superstitious beliefs, unapproved films, and advertising products that infringe Islamic principles, with certain content such as cigarette brands, national symbols, official buildings, and medical products requiring prior ministerial approval."

#TEXT = "Internet Service Providers (ISPs) and Internet Content Providers (ICPs) must obtain a Class Licence from the Minister before operating, register within 14 days of commencing operations, comply with license conditions, and assist in investigations by providing requested information."

#TEXT = "Additionally, the Cybersecurity Order CAP.272, No.S20, 2023 grant authorities powers to monitor, restrict, or remove content that threatens national security, public safety, or critical infrastructure, enabling broad government control over online activities."

# ============================================================
# 🟦 Test model on RDTII guide
# ============================================================

#pillar_5_Telecommunications regulations and competition

#indicator_1_Lack of passive infrastructure sharing
#TEXT = "Thailand requires telecom operators in both fixed and mobile sectors to share their wireless telecommunication network infrastructures, including tower/mast, site, building and method_combine_dataset physical facilities, with method_combine_dataset telecom operators in a reasonable and fair manner, without discrimination."

#TEXT = "Pakistan, Vanuatu, and Hong Kong (China) do not mandate a passive sharing obligation. However, infrastructure sharing is effectively implemented based on a commercial agreement in both mobile and fixed sectors. New Zealand does not require sharing obligation; however, a framework for voluntary co-location of cellular infrastructure is in place to facilitate sharing of towers, masts and related sites."

#indicator_3_Shares owned by the Government
#TEXT = "A de jure monopoly (statutory monopoly) is in place when telecommunications networks and services must be exclusively state-owned. For example, Turkmenistan’s Law on Communication mandates exclusive state ownership of telecom networks used for national security and national broadcasting. Turkmentelekom TC and Altyn Asyr CJSC (TMcell) are 100% state-owned."


#Pillar 6. Cross-border data policies
#TEXT = "Additionally, the Cybersecurity Order CAP.272, No.S20, 2023 grant authorities powers to monitor, restrict, or remove content that threatens national security, public safety, or critical infrastructure, enabling broad government control over online activities."

#TEXT = "Brazil sets out regulations on how financial institutions and method_combine_dataset institutions regulated by the Brazilian Central Bank should hire cloud computing services from providers that store or process information outside Brazil."

#TEXT = "Chile requires banking institutions to outsource data processing services outside the country to have a contingency data processing centre located within the country."

# TEXT = ''' (1)
# Applications for permits to import shall require specific information
# including
#
# (a)  Full name, residential address and postal address of the importer;
#
# (b)  Name and address of exporter or persons from whom plant or plant material
# will be obtained;
#
# (c)  Quantity and name (botanical name if approximate) of all material proposed
# to import;
#
# (d)  Mode of transport, point of entry and approximate date of arrival.
#
# (2)
# In granting any permits the Director shall give approval to import and
# indicate conditions to be met to satisfy quarantine requirements. '''


# ============================================================
# 🟦 Test the model on economic brief of Brunei
# ============================================================
#pillar4_Intellectual Property Rights
#TEXT = "Under the Patent Rules No. S19/2012 and Patents Order No. S57/2011, Brunei Darussalam applies a transparent, non-discriminatory patent-filing process through authorized local agents, while residents wishing to seek patents abroad must obtain prior approval from the Brunei Registry of Patents and wait two months after filing locally."

#TEXT = "Robust civil and administrative procedures, remedies, and provisional measures are readily available for patent and copyright enforcement, including injunctions, damages, destruction of infringing goods, and seizure of infringing copies."

#TEXT = "Brunei Darussalam is a contracting state to the Patent Cooperation Treaty, facilitating streamlined international patent filings. Based on fair dealing provisions in Sections 33 and 34 of the Emergency (Copyright) Order No.S14/2000 and Copyright (Amendment) Order No.S92/2013, clear copyright exceptions exist for research, private study, criticism, reviews, and news reporting."

#4_8 and
#TEXT = "Brunei Darussalam is a signatory to the WIPO Copyright Treaty and has acceded to the WIPO Performances and Phonograms Treaty. "

#TEXT = "There are no requirements to disclose source code, algorithms, or method_combine_dataset trade secrets, which are protected under common law principles and Article 39 of the Agreement on Trade-Related Aspects of Intellectual Property Rights (TRIPS)."
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
#TEXT = "Sub-section VII (of the National Preference) of the New Public Procurement Code of Decree No. 2018/366 imposes restrictions for participation of foreign companies in public procurement. Art. 106 states that: When awarding a contract within the framework of an international consultation, a margin of preference is granted, to equivalent offers and in order of priority to the tenders presented by: a) a natural person of Cameroonian nationality or a legal person under Cameroonian law; b) a company whose capital is wholly or mainly held by persons of Cameroonian nationality; c) a natural person or a legal person justifying an economic activity on the territory of Cameroon; d) a group of companies associating Cameroonian companies; Tenders are considered equivalent when they have fulfilled the required technical conditions; For quantifiable works and services contracts, the national preference margin is 10% for the companies referred to in paragraph 1 above; For supply contracts, the national preference criterion can only be taken into account if the supply undergoes a transformation at the local or regional level of at least 15%;There is no national preference for non-quantifiable service contracts including intellectual services;National preference can only be applied when the tender documents provide for it."


#Cameroon 4_5
#TEXT = "Cameroon has a clear regime of copyright exceptions, which enable the lawful use of copyrighted work by others without obtaining permission (Art. 29 of Law No. 2000/011). The regime follows the fair use model. In addition, the Revised Bangui Agreement, which is a regional intellectual property law that is not only a regional convention applicable in all member states but also serves as a national intellectual property law in Cameroon and each of the method_combine_dataset member states, contains provisions on copyright in Annex VII: Chapter IV (Limitations to Economic Rights) and establishes a regime of copyright exceptions."


#DRC 5_5 / 9_1
#TEXT = "With the Law No. 013/2002 of 2002, which governs the telecommunications sector, the government is given the power to take over the means of communication in the interest of national security. Art. 46 provides that the State may prohibit the use of telecommunication facilities, in full or in part, for any period of time, as it deems fit, in the interests of public security or national defence, the public telecommunications service, or for any method_combine_dataset reason. "

#TEXT = "In June 2018, the Minister of Posts, Telecommunications and Information, signed a Ministerial Order strengthening the control of online media (Ministerial Order No 011/CAB/M-CM/LOM/2018)."

#Algeria_2_3
#TEXT = "According to Art. 5 of Presidential Decree No. 15-247, in order to ensure the efficiency of the order and the proper use of public funds, public contracts must respect the principles of free access to public procurement, equal treatment of candidates regardless of nationality, and transparency of procedures. However, Art. 85 of the Presidential Decree also requires that the governmental purchaser, in the process of drafting the eligibility criteria and the offer evaluation procedure, must consider whether national production and/or national entities can meet the needs of the contract – and therefore by that extent also favour batches or products that can be subcontracted or acquired on the local Algerian market. In circumstances when relying on national production and/or entities is not possible, foreign companies can bid with certain limitations. Imported products may only be used in circumstances when the equivalent of the products is either not available at a national level or the national products do not possess the acceptable quality standards."

# ============================================================
# PREDICT FUNCTION
# ============================================================

def predict(text):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)

        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

    return pred_id, confidence

# ============================================================
# RUN PREDICTION
# ============================================================

pred_id, conf = predict(TEXT)
label = id2label[pred_id]

print("\n=======================")
print("🔮 Prediction Result")
print("=======================\n")
print(f"Input text: {TEXT}")
print(f"Predicted label: {label}")
print(f"Confidence: {conf:.4f}")
print("\n=======================")
