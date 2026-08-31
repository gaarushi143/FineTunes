"""
Generate synthetic training data for the Invoice & Receipt Auditor fine-tuning exercise.

Produces ~50 training examples + ~10 test examples in OpenAI JSONL format.
Each example: a messy receipt in → structured JSON audit out.
"""

import json
import random
import os

random.seed(42)

SYSTEM_PROMPT = (
    "You are an expense receipt auditor. Extract structured data from raw receipts "
    "and flag policy violations. Respond only with valid JSON."
)

# ----- Expense Policy Rules (baked into training data) -----
# - Daily meal limit: $75 per person
# - Tips must not exceed 20% of subtotal
# - Alcohol is not reimbursable
# - Personal items are not reimbursable
# - Receipts must be itemized (totals-only get flagged)
# - Weekend/holiday expenses need manager pre-approval
# - Hotel nightly rate cap: $250
# - Software subscriptions need IT approval above $50/month
# - Rideshare/taxi single-trip cap: $75

# ----- Receipt templates -----

def restaurant_clean():
    vendors = [
        ("Olive Garden", "Italian"),
        ("Chipotle", "Mexican"),
        ("Panda Express", "Chinese"),
        ("Panera Bread", "Bakery-Cafe"),
        ("The Capital Grille", "Fine Dining"),
        ("Sweetgreen", "Salads"),
    ]
    vendor, cuisine = random.choice(vendors)
    date = f"2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    weekday = random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    is_weekend = weekday in ("Saturday", "Sunday")

    items = []
    item_pool = [
        ("Caesar Salad", round(random.uniform(10, 16), 2)),
        ("Grilled Chicken Sandwich", round(random.uniform(12, 18), 2)),
        ("Pasta Primavera", round(random.uniform(14, 22), 2)),
        ("Steak Entree", round(random.uniform(28, 45), 2)),
        ("Soup of the Day", round(random.uniform(6, 10), 2)),
        ("Iced Tea", round(random.uniform(3, 5), 2)),
        ("Coffee", round(random.uniform(3, 6), 2)),
    ]
    chosen = random.sample(item_pool, k=random.randint(2, 4))
    for name, price in chosen:
        items.append({"name": name, "amount": price})

    subtotal = round(sum(i["amount"] for i in items), 2)
    tax = round(subtotal * random.uniform(0.06, 0.10), 2)

    include_alcohol = random.random() < 0.25
    if include_alcohol:
        drink = random.choice(["Glass of Wine", "Beer", "Margarita", "Cocktail"])
        drink_price = round(random.uniform(8, 18), 2)
        items.append({"name": drink, "amount": drink_price})
        subtotal = round(subtotal + drink_price, 2)

    tip_pct = random.choice([0.15, 0.18, 0.20, 0.22, 0.25])
    tip = round(subtotal * tip_pct, 2)
    total = round(subtotal + tax + tip, 2)

    names = ["Sarah M.", "John D.", "Priya K.", "Mike T.", "Lisa W.", "Carlos R.", "Emily Z."]
    depts = ["Marketing", "Engineering", "Sales", "Finance", "HR", "Operations"]
    submitter = random.choice(names)
    dept = random.choice(depts)

    receipt_text = f"{vendor}\n{date} ({weekday})\n"
    for i in items:
        receipt_text += f"  {i['name']}: ${i['amount']:.2f}\n"
    receipt_text += f"Subtotal: ${subtotal:.2f}\nTax: ${tax:.2f}\nTip: ${tip:.2f}\nTotal: ${total:.2f}\n"
    receipt_text += f"Submitted by: {submitter}, {dept} dept\nCategory: Business Meal"

    flags = []
    if total > 75:
        flags.append({"rule": "MEAL_OVER_LIMIT", "severity": "warning", "detail": f"Total ${total:.2f} exceeds $75 daily meal limit"})
    if include_alcohol:
        flags.append({"rule": "ALCOHOL_PRESENT", "severity": "violation", "detail": f"{drink} (${drink_price:.2f}) is not reimbursable"})
    if tip_pct > 0.20:
        flags.append({"rule": "TIP_OVER_20_PCT", "severity": "warning", "detail": f"Tip is {tip_pct*100:.0f}% of subtotal, exceeds 20% guideline"})
    if is_weekend:
        flags.append({"rule": "WEEKEND_EXPENSE", "severity": "warning", "detail": "Weekend expense requires manager pre-approval"})

    if any(f["severity"] == "violation" for f in flags):
        verdict = "REJECTED"
    elif flags:
        verdict = "REVIEW"
    else:
        verdict = "APPROVED"

    audit_output = {
        "vendor": vendor,
        "date": date,
        "total": total,
        "items": items + [{"name": "Tax", "amount": tax}, {"name": "Tip", "amount": tip}],
        "submitter": submitter,
        "department": dept,
        "category": "Business Meal",
        "flags": flags,
        "verdict": verdict,
    }

    return receipt_text, audit_output


def restaurant_ocr_messy():
    receipt, audit = restaurant_clean()
    replacements = {"o": "0", "O": "0", "l": "1", "I": "1", "S": "$", "a": "@"}
    messy = ""
    for ch in receipt:
        if random.random() < 0.08 and ch in replacements:
            messy += replacements[ch]
        elif random.random() < 0.03:
            messy += ""  # drop char
        else:
            messy += ch
    messy = messy.replace("\n  ", "\n ").replace("Subtotal", "Subt0tal").replace("Total", "T0t@l")
    return messy, audit


def rideshare():
    vendors = ["Uber", "Lyft", "Via"]
    vendor = random.choice(vendors)
    date = f"2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    weekday = random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])

    routes = [
        ("Airport", "Downtown Hotel"),
        ("Office", "Client Site"),
        ("Home", "Airport"),
        ("Conference Center", "Office"),
        ("Train Station", "Office"),
        ("Hotel", "Restaurant"),
    ]
    origin, dest = random.choice(routes)
    distance = round(random.uniform(5, 35), 1)
    fare = round(random.uniform(15, 85), 2)
    tip = round(fare * random.choice([0.0, 0.15, 0.18, 0.20, 0.25]), 2)
    total = round(fare + tip, 2)

    names = ["Alex P.", "Jordan K.", "Taylor S.", "Morgan B.", "Casey L."]
    depts = ["Sales", "Engineering", "Marketing", "Executive"]
    submitter = random.choice(names)
    dept = random.choice(depts)

    receipt_text = (
        f"{vendor} Ride\n{date} ({weekday})\n"
        f"{origin} -> {dest}\n"
        f"Distance: {distance} mi\n"
        f"Fare: ${fare:.2f}  Tip: ${tip:.2f}  Total: ${total:.2f}\n"
        f"Submitted by: {submitter}, {dept}\nCategory: Transportation"
    )

    flags = []
    if total > 75:
        flags.append({"rule": "RIDESHARE_OVER_LIMIT", "severity": "warning", "detail": f"Total ${total:.2f} exceeds $75 single-trip cap"})
    if tip > 0 and tip / fare > 0.20:
        flags.append({"rule": "TIP_OVER_20_PCT", "severity": "warning", "detail": f"Tip is {tip/fare*100:.0f}% of fare, exceeds 20% guideline"})

    verdict = "REVIEW" if flags else "APPROVED"

    audit_output = {
        "vendor": vendor,
        "date": date,
        "total": total,
        "items": [{"name": "Ride fare", "amount": fare}, {"name": "Tip", "amount": tip}],
        "submitter": submitter,
        "department": dept,
        "category": "Transportation",
        "flags": flags,
        "verdict": verdict,
    }

    return receipt_text, audit_output


def hotel():
    hotels = ["Hilton Downtown", "Marriott Suites", "Holiday Inn Express", "Hyatt Regency", "Best Western Plus"]
    hotel_name = random.choice(hotels)
    checkin_month = random.randint(1, 12)
    checkin_day = random.randint(1, 25)
    nights = random.randint(1, 4)
    checkout_day = checkin_day + nights
    date = f"2026-{checkin_month:02d}-{checkin_day:02d}"
    rate = random.choice([149, 189, 199, 229, 259, 289, 319])
    room_total = rate * nights
    taxes = round(room_total * 0.13, 2)
    has_minibar = random.random() < 0.3
    minibar_charge = round(random.uniform(8, 35), 2) if has_minibar else 0
    total = round(room_total + taxes + minibar_charge, 2)

    names = ["David H.", "Rachel N.", "Tom W.", "Anita S.", "James C."]
    depts = ["Sales", "Executive", "Engineering", "Marketing"]
    submitter = random.choice(names)
    dept = random.choice(depts)

    receipt_text = (
        f"{hotel_name}\n"
        f"Check-in: {checkin_month:02d}/{checkin_day:02d}/2026  Check-out: {checkin_month:02d}/{checkout_day:02d}/2026\n"
        f"Room: Standard King, {nights} night(s) @ ${rate}/night\n"
        f"Room charges: ${room_total:.2f}\n"
        f"Taxes & fees: ${taxes:.2f}\n"
    )
    if has_minibar:
        receipt_text += f"Minibar: ${minibar_charge:.2f}\n"
    receipt_text += (
        f"Total: ${total:.2f}\n"
        f"Submitted by: {submitter}, {dept}\nCategory: Lodging"
    )

    flags = []
    if rate > 250:
        flags.append({"rule": "HOTEL_OVER_NIGHTLY_CAP", "severity": "warning", "detail": f"Nightly rate ${rate} exceeds $250 cap"})
    if has_minibar:
        flags.append({"rule": "PERSONAL_ITEM", "severity": "warning", "detail": f"Minibar charge (${minibar_charge:.2f}) may be personal expense"})

    if any(f["severity"] == "violation" for f in flags):
        verdict = "REJECTED"
    elif flags:
        verdict = "REVIEW"
    else:
        verdict = "APPROVED"

    items = [{"name": f"Room ({nights} nights)", "amount": room_total}, {"name": "Taxes & fees", "amount": taxes}]
    if has_minibar:
        items.append({"name": "Minibar", "amount": minibar_charge})

    audit_output = {
        "vendor": hotel_name,
        "date": date,
        "total": total,
        "items": items,
        "submitter": submitter,
        "department": dept,
        "category": "Lodging",
        "flags": flags,
        "verdict": verdict,
    }

    return receipt_text, audit_output


def office_supplies():
    stores = ["Staples", "Office Depot", "Amazon.com", "Best Buy"]
    store = random.choice(stores)
    date = f"2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}"

    supply_pool = [
        ("Printer Paper (5 ream)", round(random.uniform(25, 40), 2)),
        ("Ink Cartridge", round(random.uniform(28, 55), 2)),
        ("Wireless Mouse", round(random.uniform(19, 45), 2)),
        ("USB-C Hub", round(random.uniform(25, 50), 2)),
        ("Desk Lamp", round(random.uniform(22, 45), 2)),
        ("Notebook (3-pack)", round(random.uniform(8, 15), 2)),
        ("Pens (12-pack)", round(random.uniform(5, 12), 2)),
        ("Monitor Stand", round(random.uniform(30, 65), 2)),
    ]
    chosen = random.sample(supply_pool, k=random.randint(1, 4))

    has_personal = random.random() < 0.25
    if has_personal:
        personal_item = random.choice([
            ("Phone Case (personal)", round(random.uniform(15, 35), 2)),
            ("Bluetooth Speaker", round(random.uniform(25, 60), 2)),
            ("Snack Box", round(random.uniform(12, 25), 2)),
        ])
        chosen.append(personal_item)

    items = [{"name": name, "amount": price} for name, price in chosen]
    subtotal = round(sum(price for _, price in chosen), 2)
    tax = round(subtotal * random.uniform(0.06, 0.10), 2)
    total = round(subtotal + tax, 2)

    names = ["Kevin R.", "Linda M.", "Sarah M.", "Brian T.", "Amy L."]
    depts = ["Operations", "Engineering", "Marketing", "HR", "Finance"]
    submitter = random.choice(names)
    dept = random.choice(depts)

    receipt_text = f"{store}\nOrder Date: {date}\n"
    for name, price in chosen:
        receipt_text += f"  {name} — ${price:.2f}\n"
    receipt_text += f"Subtotal: ${subtotal:.2f}\nTax: ${tax:.2f}\nTotal: ${total:.2f}\n"
    receipt_text += f"Submitted by: {submitter}, {dept}\nCategory: Office Supplies"

    flags = []
    if has_personal:
        p_name, p_price = personal_item
        flags.append({"rule": "PERSONAL_ITEM", "severity": "violation", "detail": f"{p_name} (${p_price:.2f}) appears to be a personal purchase"})

    items.append({"name": "Tax", "amount": tax})

    if any(f["severity"] == "violation" for f in flags):
        verdict = "REJECTED"
    elif flags:
        verdict = "REVIEW"
    else:
        verdict = "APPROVED"

    audit_output = {
        "vendor": store,
        "date": date,
        "total": total,
        "items": items,
        "submitter": submitter,
        "department": dept,
        "category": "Office Supplies",
        "flags": flags,
        "verdict": verdict,
    }

    return receipt_text, audit_output


def software_subscription():
    subs = [
        ("Figma", "Design Tool"),
        ("Slack", "Communication"),
        ("Notion", "Productivity"),
        ("GitHub", "Developer Tools"),
        ("Zoom", "Video Conferencing"),
        ("Adobe Creative Cloud", "Design Tool"),
        ("Jira", "Project Management"),
        ("1Password Teams", "Security"),
    ]
    name, stype = random.choice(subs)
    date = f"2026-{random.randint(1,12):02d}-01"
    plan = random.choice(["monthly", "annual"])
    monthly_cost = round(random.uniform(10, 85), 2)
    if plan == "annual":
        amount = round(monthly_cost * 12 * 0.8, 2)
        period = "annual"
    else:
        amount = monthly_cost
        period = "monthly"

    names = ["Chris E.", "Dana F.", "Pat V.", "Robin S."]
    depts = ["Engineering", "Design", "Product", "IT"]
    submitter = random.choice(names)
    dept = random.choice(depts)

    receipt_text = (
        f"{name} — {stype}\n"
        f"Invoice Date: {date}\n"
        f"Plan: {plan.title()} ({period})\n"
        f"Amount: ${amount:.2f}\n"
        f"Submitted by: {submitter}, {dept}\nCategory: Software Subscription"
    )

    flags = []
    effective_monthly = monthly_cost if plan == "monthly" else round(amount / 12, 2)
    if effective_monthly > 50:
        flags.append({"rule": "SOFTWARE_OVER_50_MONTHLY", "severity": "warning", "detail": f"Effective monthly cost ~${effective_monthly:.2f} exceeds $50/month threshold — needs IT approval"})

    verdict = "REVIEW" if flags else "APPROVED"

    audit_output = {
        "vendor": name,
        "date": date,
        "total": amount,
        "items": [{"name": f"{name} {plan.title()} Plan", "amount": amount}],
        "submitter": submitter,
        "department": dept,
        "category": "Software Subscription",
        "flags": flags,
        "verdict": verdict,
    }

    return receipt_text, audit_output


def minimal_receipt():
    """Very sparse receipts that test extraction from minimal info."""
    templates = [
        ("uber $23.50 airport run 8/14", {
            "vendor": "Uber", "date": "2026-08-14", "total": 23.50,
            "items": [{"name": "Ride", "amount": 23.50}],
            "submitter": "Unknown", "department": "Unknown",
            "category": "Transportation",
            "flags": [{"rule": "MISSING_SUBMITTER", "severity": "warning", "detail": "No submitter identified on receipt"}],
            "verdict": "REVIEW"
        }),
        ("starbucks 2x latte $12.40 — jess, mktg", {
            "vendor": "Starbucks", "date": None, "total": 12.40,
            "items": [{"name": "Latte x2", "amount": 12.40}],
            "submitter": "Jess", "department": "Marketing",
            "category": "Business Meal",
            "flags": [{"rule": "MISSING_DATE", "severity": "warning", "detail": "No date found on receipt"}],
            "verdict": "REVIEW"
        }),
        ("gas station fill up $48.00 company car plate XYZ-123. mike, ops", {
            "vendor": "Gas Station", "date": None, "total": 48.00,
            "items": [{"name": "Fuel", "amount": 48.00}],
            "submitter": "Mike", "department": "Operations",
            "category": "Transportation",
            "flags": [{"rule": "MISSING_DATE", "severity": "warning", "detail": "No date found on receipt"}],
            "verdict": "REVIEW"
        }),
        ("FedEx shipping 2 boxes to client, $34.75, 09/03/2026, ops team", {
            "vendor": "FedEx", "date": "2026-09-03", "total": 34.75,
            "items": [{"name": "Shipping (2 boxes)", "amount": 34.75}],
            "submitter": "Unknown", "department": "Operations",
            "category": "Shipping",
            "flags": [{"rule": "MISSING_SUBMITTER", "severity": "warning", "detail": "No specific submitter identified"}],
            "verdict": "REVIEW"
        }),
        ("parking garage downtown 4hrs $22.00 tues. alex, sales", {
            "vendor": "Parking Garage", "date": None, "total": 22.00,
            "items": [{"name": "Parking (4 hours)", "amount": 22.00}],
            "submitter": "Alex", "department": "Sales",
            "category": "Transportation",
            "flags": [{"rule": "MISSING_DATE", "severity": "warning", "detail": "No specific date found on receipt"}],
            "verdict": "REVIEW"
        }),
    ]
    receipt_text, audit_output = random.choice(templates)
    return receipt_text, audit_output


def food_delivery():
    vendors = ["DoorDash", "Uber Eats", "Grubhub", "Postmates"]
    vendor = random.choice(vendors)
    date = f"2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    weekday = random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    is_weekend = weekday in ("Saturday", "Sunday")

    food_items = [
        ("Burrito Bowl", round(random.uniform(10, 15), 2)),
        ("Pad Thai", round(random.uniform(13, 18), 2)),
        ("Pizza (Large)", round(random.uniform(16, 24), 2)),
        ("Sushi Combo", round(random.uniform(18, 30), 2)),
        ("Burger & Fries", round(random.uniform(12, 18), 2)),
        ("Salad", round(random.uniform(10, 14), 2)),
    ]
    chosen = random.sample(food_items, k=random.randint(2, 4))
    items = [{"name": n, "amount": p} for n, p in chosen]
    subtotal = round(sum(p for _, p in chosen), 2)

    delivery_fee = round(random.uniform(3, 8), 2)
    service_fee = round(subtotal * 0.15, 2)
    tip_pct = random.choice([0.15, 0.18, 0.20, 0.22, 0.30])
    tip = round(subtotal * tip_pct, 2)
    total = round(subtotal + delivery_fee + service_fee + tip, 2)

    items.extend([
        {"name": "Delivery fee", "amount": delivery_fee},
        {"name": "Service fee", "amount": service_fee},
        {"name": "Tip", "amount": tip},
    ])

    names = ["Priya K.", "John D.", "Sarah M.", "Carlos R.", "Emily Z."]
    depts = ["Engineering", "Marketing", "Sales", "Product"]
    submitter = random.choice(names)
    dept = random.choice(depts)

    receipt_text = (
        f"{vendor} Order\n{date} ({weekday})\n"
    )
    for n, p in chosen:
        receipt_text += f"  {n}: ${p:.2f}\n"
    receipt_text += (
        f"Delivery fee: ${delivery_fee:.2f}\n"
        f"Service fee: ${service_fee:.2f}\n"
        f"Tip: ${tip:.2f}\n"
        f"Total: ${total:.2f}\n"
        f"Submitted by: {submitter}, {dept}\nCategory: Team Lunch"
    )

    flags = []
    if total > 75:
        flags.append({"rule": "MEAL_OVER_LIMIT", "severity": "warning", "detail": f"Total ${total:.2f} exceeds $75 daily meal limit"})
    if tip_pct > 0.20:
        flags.append({"rule": "TIP_OVER_20_PCT", "severity": "warning", "detail": f"Tip is {tip_pct*100:.0f}% of subtotal, exceeds 20% guideline"})
    if is_weekend:
        flags.append({"rule": "WEEKEND_EXPENSE", "severity": "warning", "detail": "Weekend expense requires manager pre-approval"})

    if any(f["severity"] == "violation" for f in flags):
        verdict = "REJECTED"
    elif flags:
        verdict = "REVIEW"
    else:
        verdict = "APPROVED"

    audit_output = {
        "vendor": vendor,
        "date": date,
        "total": total,
        "items": items,
        "submitter": submitter,
        "department": dept,
        "category": "Team Lunch",
        "flags": flags,
        "verdict": verdict,
    }

    return receipt_text, audit_output


# ----- Assemble the dataset -----

def make_example(receipt_text, audit_output):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": receipt_text},
            {"role": "assistant", "content": json.dumps(audit_output)},
        ]
    }


def generate_dataset():
    examples = []

    # 10 clean restaurant receipts
    for _ in range(10):
        receipt, audit = restaurant_clean()
        examples.append(make_example(receipt, audit))

    # 5 OCR-messy restaurant receipts
    for _ in range(5):
        receipt, audit = restaurant_ocr_messy()
        examples.append(make_example(receipt, audit))

    # 8 rideshare receipts
    for _ in range(8):
        receipt, audit = rideshare()
        examples.append(make_example(receipt, audit))

    # 8 hotel receipts
    for _ in range(8):
        receipt, audit = hotel()
        examples.append(make_example(receipt, audit))

    # 8 office supply receipts
    for _ in range(8):
        receipt, audit = office_supplies()
        examples.append(make_example(receipt, audit))

    # 6 software subscription receipts
    for _ in range(6):
        receipt, audit = software_subscription()
        examples.append(make_example(receipt, audit))

    # 5 minimal/sparse receipts
    for _ in range(5):
        receipt, audit = minimal_receipt()
        examples.append(make_example(receipt, audit))

    # 8 food delivery receipts
    for _ in range(8):
        receipt, audit = food_delivery()
        examples.append(make_example(receipt, audit))

    random.shuffle(examples)
    return examples


def main():
    examples = generate_dataset()

    # Split: first 50 for training, rest for testing
    train = examples[:50]
    test = examples[50:]

    # If we have fewer than 8 test examples, move some from train
    if len(test) < 8:
        extra_needed = 8 - len(test)
        test.extend(train[-extra_needed:])
        train = train[:-extra_needed]

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(test_path, "w") as f:
        for ex in test:
            f.write(json.dumps(ex) + "\n")

    # Summary stats
    verdicts = {}
    categories = {}
    for ex in train:
        audit = json.loads(ex["messages"][2]["content"])
        v = audit["verdict"]
        c = audit["category"]
        verdicts[v] = verdicts.get(v, 0) + 1
        categories[c] = categories.get(c, 0) + 1

    print(f"Generated {len(train)} training examples -> {train_path}")
    print(f"Generated {len(test)} test examples -> {test_path}")
    print(f"\nVerdict distribution (train): {dict(sorted(verdicts.items()))}")
    print(f"Category distribution (train): {dict(sorted(categories.items()))}")


if __name__ == "__main__":
    main()
