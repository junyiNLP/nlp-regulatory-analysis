import json
from pathlib import Path
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# PATH CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "stage1_training_4_multilabel"
MODEL_DIR = TRAIN_DIR / "model_training_4_multi"

# Dataset is here (NOT inside model folder)
DATA_FILE = TRAIN_DIR / "unified_ds_full.csv"

# ============================================================
# LOAD DATASET → RECONSTRUCT LABEL MAP
# ============================================================
print("🔹 Reconstructing label mapping from dataset...")

df = pd.read_csv(DATA_FILE)

# Convert label column to categorical for exact training order
df["label"] = df["label"].astype("category")
label_list = list(df["label"].cat.categories)

# Build id2label mapping: index → indicator label
id2label = {i: label_list[i] for i in range(len(label_list))}

print(f"✅ Loaded {len(id2label)} labels.")
print("Example mapping:", id2label[0], ",", id2label[1])

# ============================================================
# LOAD MODEL + TOKENIZER
# ============================================================
print("\n🔹 Loading Multi-label model...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.eval()

# Device selection
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("🔹 Using device:", device)
model.to(device)

# ============================================================
# MULTI-LABEL PREDICTION FUNCTION
# ============================================================
def predict_multilabel(text, top_k=5):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.sigmoid(logits)[0]

    probs = probs.cpu()

    # Top-k predicted indices
    top_probs, top_ids = torch.topk(probs, k=top_k)

    results = []
    for p, idx in zip(top_probs, top_ids):
        real_label = id2label[int(idx)]     # Correct mapped indicator
        results.append((real_label, float(p)))

    return results

# ============================================================
# TEST PREDICTION
# ============================================================

# TEXT = "With the Law No. 013/2002 of 2002, which governs the telecommunications sector, the government is given the power to take over the means of communication in the interest of national security. Art. 46 provides that the State may prohibit the use of telecommunication facilities, in full or in part, for any period of time, as it deems fit, in the interests of public security or national defence, the public telecommunications service, or for any method_combine_dataset reason. "


# TEXT = "Brunei Darussalam is a signatory to the WIPO Copyright Treaty and has acceded to the WIPO Performances and Phonograms Treaty. "

# TEXT = ''' 4.1 Application of Part 4
# This Part sets out the rules with which a carriage service provider must
# comply in relation to obtaining identifying information from, and verifying
# the identity of, a customer of a prepaid mobile carriage service for the
# purposes of paragraph 2.3(1)(a).
# 4.2 (1) (2) Requirements to be satisfied before service is activated
# Subject to subsection (2), the carriage service provider must not activate the
# prepaid mobile carriage service unless the provider has:
# (a) obtained information from the customer in accordance with section
# 4.3; and
# (b) verified the identity of the customer in accordance with section 4.4 or
# 4.5, whichever is applicable.
# The rule in subsection (1) does not apply where:
# (a) the carriage service provider has activated the prepaid mobile carriage
# service at the same time as, or immediately before, the provider
# receives confirmation that the service activator’s specified financial
# account is active in accordance with paragraph (2)(c) and subitem (4)
# of item 4 of Schedule 1; or
# (b) the carriage service provider has temporarily activated the prepaid
# mobile carriage service in accordance with paragraph (2)(c) of item 5
# of Schedule 1.) '''

# TEXT = '''Robust civil and administrative procedures, remedies, and provisional measures are available for
# patent enforcement in Thailand under the Patent Act 16 , which provides penalties including
# imprisonment or fines for infringements. Furthermore, Thailand has been a contracting state to the
# WIPO Patent Cooperation Treaty since 2009, facilitating streamlined international patent filings.
# Meanwhile, the Copyright Act No. 5 B.E. 2565/2022 provides exceptions for research, education,
# personal use, and method_combine_dataset non-profit purposes, and is supported by enforcement procedures and notice-
# and-takedown mechanisms. '''

# TEXT = '''These measures are also complemented by the Computer-Related Crime
# Act B.E. 2550/2007. Moreover, Thailand acceded to the WIPO Copyright Treaty in 2022 and is
# amending its law to align with the WIPO Performances and Phonograms Treaty. In the economy, trade
# secrets are protected under the Trade Secrets Act B.E. 2545/2002, although certain government-
# authorized disclosures for public health or method_combine_dataset public interests may occur without the owner’s
# consent17. '''

# TEXT = '''Nonetheless, there are restrictions on patent applications. Under the Patent Act, patent filing
# in Thailand is limited to Thai nationals, juristic persons headquartered in Thailand, nationals of
# countries party to international patent conventions to which Thailand is also a party, or persons with
# domicile or commercial establishments in Thailand or in convention countries. Non-resident applicants
# must file through registered agents as required by Ministerial Regulation No. 21 issued under the
# Patent Act.'''


# TEXT = ''' In Thailand, foreign participation in telecom services is governed by the Foreign Business Act
# B.E.2542/1999 and the Telecommunications Business Act B.E. 2544/2001, with foreign ownership
# capped at 49% for Type 2 and Type 3 licenses24, while Type 1 license holders face no restrictions, and
# foreign investors must obtain a foreign business license. The telecommunication sector is partially
# dominated by state-owned enterprises, such as National Telecom, which holds 66% of the fixed-line
# market, whereas major mobile providers are private. In Thailand, Operators with significant market
# power must submit accounting separation reports under NBTC Notification B.E. 2565/202225, though
# functional separation is not required. Meanwhile, under the NBTC Notifications, operators with
# significant market power in satellite services face stringent licensing conditions, creating significant
# entry barriers.26 On a positive note, Thailand has adopted the WTO Telecom Reference Paper in March
# 2022, signalling adherence to international best practices in telecom regulation. The economy
# maintains an independent regulator, the NBTC27, which supervises licensing, spectrum allocation, and
# sector oversight. Thailand promotes infrastructure sharing in both mobile and fixed-line sectors,
# lowering entry barriers and reducing network-deployment costs.'''

#TEXT = ''' A carriage service provider may verify the identity of a service activator by confirming the details of the service activator’s existing eligible prepaid (direct debit) account with the carriage service provider.'''

#Algeria 8_3/7_3
# TEXT = ''' Art. 11 of Law No. 09-04 requires ICT service providers to store data allowing the identification of users of their services for a period of one year after the registration. This requirement covers all service providers, defined in Art. 2 to include any public or private entity which offers users of its services the possibility of communicating by means of a computer system and/or a telecommunications system and any method_combine_dataset entity processing or storing computer data for communication services or their users.  '''

# #Algeria 9_1
# TEXT = ''' It is reported that Algeria has a history of its government blocking sites for various reasons, including the prevention of exam cheating. Moreover, web-based Algerian media outlets have been required to be based within the country and they have been asked to inform authorities of any "illegal content". Some of the websites that have been blocked in the past include YouTube, Facebook, and Twitter, as well as news websites such as France 24 and Al-Jazeera.  '''

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

print(f"Input text: {TEXT}")

print("\n🔮 TOP-5 MULTI-LABEL PREDICTION\n")

results = predict_multilabel(TEXT, top_k=5)

for label, prob in results:
    print(f"{label:<10} | {prob:.4f}")
