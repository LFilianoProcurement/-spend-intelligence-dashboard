import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="Spend Intelligence Dashboard",
    page_icon="🔍",
    layout="wide"
)

# ── CUSTOM STYLING ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1F4E79; font-family: Arial; }
    h2, h3 { color: #2E75B6; font-family: Arial; }
    .section-divider {
        border-top: 2px solid #2E75B6;
        margin: 20px 0px;
    }
    .footer {
        text-align: center;
        color: #888888;
        font-size: 12px;
        margin-top: 40px;
        font-family: Arial;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────
BLACKLISTED_SUPPLIERS = [
    "XYZ Global Trading LLC",
    "Offshore Supply Co.",
    "Quick Parts International"
]

PCARD_LIMIT = 2500.00
PCARD_THRESHOLD_PCT = 0.85

# ── HEADER ─────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;'>🔍 Spend Intelligence Dashboard</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555; font-family:Arial;'>"
            "Procurement Fraud Detection & Compliance Monitoring  |  "
            "Louis Filiano</p>", unsafe_allow_html=True)
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# ── FILE UPLOAD ────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Upload your spend data CSV file to begin analysis",
    type="csv"
)

if uploaded_file is not None:

    # ── LOAD & CLEAN ───────────────────────────────────────
    df = pd.read_csv(uploaded_file)
    df["Amount"] = df["Amount"].str.replace("$", "", regex=False)
    df["Amount"] = df["Amount"].str.replace(",", "", regex=False).astype(float)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Product_Description"] = df["Product_Description"].fillna("")
    df["Transaction_Type"] = df["Transaction_Type"].fillna("Unknown")

    # ── BUSINESS RULES ─────────────────────────────────────

    # Rule 1: Blacklisted suppliers
    df["Blacklist_Flag"] = df["Supplier"].apply(
        lambda x: "BLACKLISTED SUPPLIER" if x in BLACKLISTED_SUPPLIERS else "")

    # Rule 2: Prohibited categories on P-Card
    df["Prohibited_Flag"] = df["Notes"].apply(
        lambda x: str(x) if "VIOLATION" in str(x) else "")

    # Rule 3: Near P-Card threshold
    df["Threshold_Flag"] = df.apply(
        lambda r: "NEAR P-CARD LIMIT"
        if (r["Transaction_Type"] == "P-Card" and
            r["Amount"] >= PCARD_LIMIT * PCARD_THRESHOLD_PCT and
            r["Amount"] <= PCARD_LIMIT)
        else "", axis=1)

    # Rule 4: P-Card splitting
    df["Split_Flag"] = ""
    pcard_df = df[df["Transaction_Type"] == "P-Card"].copy()
    for (cardholder, supplier), group in pcard_df.groupby(["Cardholder", "Supplier"]):
        group = group.sort_values("Date")
        for idx in group.index:
            window = group[
                (group["Date"] >= group.loc[idx, "Date"]) &
                (group["Date"] <= group.loc[idx, "Date"] + pd.Timedelta(days=45)) &
                (abs(group["Amount"] - group.loc[idx, "Amount"]) < 1.00)
            ]
            if len(window) >= 2:
                df.loc[window.index, "Split_Flag"] = "POTENTIAL P-CARD SPLIT"

    # Rule 5: Identical amount same supplier
    po_df = df[df["Transaction_Type"] == "Purchase Order"].copy()
    dup_mask = po_df.duplicated(subset=["Supplier", "Amount"], keep=False)
    df["Duplicate_Amount_Flag"] = ""
    df.loc[dup_mask[dup_mask].index, "Duplicate_Amount_Flag"] = "IDENTICAL AMOUNT SAME SUPPLIER"

    # Rule 6: Duplicate PO numbers
    df["Duplicate_PO_Flag"] = df.duplicated(
        subset=["PO_Number"], keep=False).apply(
        lambda x: "DUPLICATE PO NUMBER" if x else "")

    # ── AI ANOMALY DETECTION ───────────────────────────────
    le_supplier = LabelEncoder()
    le_category = LabelEncoder()
    le_type = LabelEncoder()
    df["Supplier_Code"] = le_supplier.fit_transform(df["Supplier"])
    df["Category_Code"] = le_category.fit_transform(df["Category"])
    df["Type_Code"] = le_type.fit_transform(df["Transaction_Type"])
    features = df[["Amount", "Supplier_Code", "Category_Code", "Type_Code"]]
    model = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
    model.fit(features)
    df["AI_Score"] = model.decision_function(features)
    df["AI_Flag"] = model.predict(features)

    # ── MASTER FLAG ────────────────────────────────────────
    def master_flag(row):
        flags = [
            row["Blacklist_Flag"],
            row["Prohibited_Flag"],
            row["Split_Flag"],
            row["Duplicate_Amount_Flag"],
            row["Duplicate_PO_Flag"],
            row["Threshold_Flag"],
        ]
        active = [f for f in flags if f != ""]
        if active:
            return active[0]
        if row["AI_Flag"] == -1:
            return "AI ANOMALY DETECTED"
        return "Normal"

    df["Master_Flag"] = df.apply(master_flag, axis=1)

    # ── SEGMENT DATA ───────────────────────────────────────
    all_flagged = df[df["Master_Flag"] != "Normal"]
    fraud_flags = df[df["Split_Flag"].str.len() > 0]
    compliance_flags = df[
        (df["Prohibited_Flag"].str.len() > 0) |
        (df["Threshold_Flag"].str.len() > 0)
    ]
    blacklisted = df[df["Blacklist_Flag"].str.len() > 0]
    pcard_violations = df[df["Prohibited_Flag"].str.len() > 0]
    dup_pos = df[df["Duplicate_PO_Flag"].str.len() > 0]
    identical = df[df["Duplicate_Amount_Flag"].str.len() > 0]
    near_thresh = df[df["Threshold_Flag"].str.len() > 0]

    # ══════════════════════════════════════════════════════
    # SECTION 1: EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════
    st.markdown("## 📊 Executive Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Transactions", len(df))
    c2.metric("P-Card Transactions", len(df[df["Transaction_Type"] == "P-Card"]))
    c3.metric("Purchase Orders", len(df[df["Transaction_Type"] == "Purchase Order"]))
    c4.metric("Total Spend", f"${df['Amount'].sum():,.0f}")
    c5.metric("Fraud Flags", len(all_flagged))
    c6.metric("Compliance Violations", len(compliance_flags))
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # SECTION 2: SPEND CHARTS
    # ══════════════════════════════════════════════════════
    st.markdown("## 📈 Spend Analysis")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Spend by Category")
        cat_spend = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        cat_spend.plot(kind="bar", ax=ax1, color="#2E75B6")
        ax1.set_xlabel("")
        ax1.set_ylabel("Total ($)")
        ax1.tick_params(axis='x', labelrotation=45, labelsize=8)
        plt.tight_layout()
        st.pyplot(fig1)

    with col_right:
        st.markdown("#### Spend by Transaction Type")
        type_spend = df.groupby("Transaction_Type")["Amount"].sum()
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        type_spend.plot(kind="pie", ax=ax2,
                        colors=["#2E75B6", "#C00000"],
                        autopct="%1.1f%%", startangle=90)
        ax2.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # SECTION 3: ANOMALY SCATTER
    # ══════════════════════════════════════════════════════
    st.markdown("## 🔍 Anomaly Detection Overview")
    color_map = df["Master_Flag"].apply(
        lambda x: "#C00000" if x in [
            "POTENTIAL P-CARD SPLIT", "BLACKLISTED SUPPLIER", "DUPLICATE PO NUMBER"]
        else "#FF8C00" if x in [
            "IDENTICAL AMOUNT SAME SUPPLIER", "AI ANOMALY DETECTED",
            "NEAR P-CARD LIMIT"] or "VIOLATION" in x
        else "#2E75B6"
    )
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    ax3.scatter(range(len(df)), df["Amount"],
                c=color_map, alpha=0.75,
                edgecolors="black", linewidth=0.3, s=60)
    ax3.set_title("All Transactions — Red: Fraud   Orange: Warning   Blue: Normal",
                  fontsize=12)
    ax3.set_xlabel("Transaction Index")
    ax3.set_ylabel("Amount ($)")
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    st.pyplot(fig3)
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # SECTION 4: FRAUD FLAGS
    # ══════════════════════════════════════════════════════
    st.markdown("## Fraud Detection Flags")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("P-Card Splits", len(fraud_flags))
    f2.metric("Duplicate POs", len(dup_pos))
    f3.metric("Blacklisted Suppliers", len(blacklisted))
    f4.metric("Identical Amounts", len(identical))

    if len(fraud_flags) > 0:
        st.markdown("#### P-Card Splitting Detail")
        st.dataframe(
            fraud_flags[["PO_Number", "Transaction_Type", "Supplier",
                         "Category", "Product_Description", "Amount",
                         "Date", "Cardholder", "Split_Flag"]]
            .sort_values(["Cardholder", "Date"]),
            use_container_width=True
        )

    if len(blacklisted) > 0:
        st.markdown("#### Blacklisted Supplier Transactions")
        st.dataframe(
            blacklisted[["PO_Number", "Transaction_Type", "Supplier",
                         "Category", "Amount", "Date",
                         "Cardholder", "Blacklist_Flag"]],
            use_container_width=True
        )

    if len(dup_pos) > 0:
        st.markdown("#### Duplicate PO Numbers")
        st.dataframe(
            dup_pos[["PO_Number", "Supplier", "Category",
                     "Amount", "Date", "Cardholder", "Duplicate_PO_Flag"]],
            use_container_width=True
        )

    if len(identical) > 0:
        st.markdown("#### Identical Amount Same Supplier")
        st.dataframe(
            identical[["PO_Number", "Supplier", "Category",
                       "Amount", "Date", "Cardholder", "Duplicate_Amount_Flag"]]
            .sort_values(["Supplier", "Amount"]),
            use_container_width=True
        )

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # SECTION 5: COMPLIANCE VIOLATIONS
    # ══════════════════════════════════════════════════════
    st.markdown("## Compliance Violations")
    v1, v2, v3, v4 = st.columns(4)
    chem = df[df["Prohibited_Flag"].str.contains("Chemical", na=False)]
    hazmat = df[df["Prohibited_Flag"].str.contains("Hazardous|Flammable|Controlled", na=False)]
    it_purchases = df[df["Prohibited_Flag"].str.contains("IT Purchase", na=False)]
    capital = df[df["Prohibited_Flag"].str.contains("Capital|Calibrated|PPE", na=False)]
    v1.metric("Chemical Violations", len(chem))
    v2.metric("Hazmat / Flammable", len(hazmat))
    v3.metric("Unauthorized IT", len(it_purchases))
    v4.metric("Capital / Equipment", len(capital))

    if len(pcard_violations) > 0:
        st.markdown("#### Prohibited P-Card Purchases")
        st.dataframe(
            pcard_violations[["PO_Number", "Transaction_Type", "Supplier",
                              "Category", "Product_Description", "Amount",
                              "Date", "Cardholder", "Prohibited_Flag"]]
            .sort_values("Category"),
            use_container_width=True
        )

    if len(near_thresh) > 0:
        st.markdown("#### Transactions Near P-Card Limit")
        st.dataframe(
            near_thresh[["PO_Number", "Supplier", "Category",
                         "Amount", "Date", "Cardholder", "Threshold_Flag"]],
            use_container_width=True
        )

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # SECTION 6: CARDHOLDER RISK SUMMARY
    # ══════════════════════════════════════════════════════
    st.markdown("## Cardholder Risk Summary")
    st.markdown("Cardholders with multiple flag types warrant priority review.")
    cardholder_flags = all_flagged.groupby("Cardholder")["Master_Flag"]\
        .count().reset_index()
    cardholder_flags.columns = ["Cardholder", "Total Flags"]
    cardholder_spend = df.groupby("Cardholder")["Amount"]\
        .sum().reset_index()
    cardholder_spend.columns = ["Cardholder", "Total Spend"]
    cardholder_summary = cardholder_flags.merge(
        cardholder_spend, on="Cardholder").sort_values(
        "Total Flags", ascending=False)
    cardholder_summary["Total Spend"] = cardholder_summary["Total Spend"]\
        .apply(lambda x: f"${x:,.2f}")
    st.dataframe(cardholder_summary, use_container_width=True)
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # SECTION 7: DOWNLOAD
    # ══════════════════════════════════════════════════════
    st.markdown("## Download Full Results")
    output = df[["PO_Number", "Transaction_Type", "Supplier", "Category",
                 "Product_Description", "Amount", "Date", "Cardholder",
                 "Master_Flag", "Split_Flag", "Blacklist_Flag",
                 "Prohibited_Flag", "Threshold_Flag",
                 "Duplicate_PO_Flag", "Duplicate_Amount_Flag"]].copy()
    csv_out = output.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download Complete Analysis as CSV",
        data=csv_out,
        file_name="spend_intelligence_results.csv",
        mime="text/csv"
    )

    st.markdown(
        "<div class='footer'>Spend Intelligence Dashboard  |  "
        "Louis Filiano  |  Powered by Python & Scikit-Learn</div>",
        unsafe_allow_html=True
    )

else:
    st.info("Upload your spend data CSV file above to begin analysis.")
    st.markdown("""
    **This dashboard detects:**
    - P-Card transaction splitting (threshold avoidance fraud)
    - Blacklisted or unauthorized supplier usage
    - Duplicate PO numbers (double billing)
    - Identical amounts from the same supplier
    - Transactions near the P-Card approval limit
    - Prohibited commodity purchases on P-Card
    - Unauthorized IT, chemical, and capital equipment purchases
    - AI-powered anomaly detection on all transactions
    """)
