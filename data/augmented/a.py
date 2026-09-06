# Full recreation of html_rows and HTML generation
reviews_data = [
    {"text": "Super greasy pizza which usually I don't mind but it was salty and greasy! I wanted to order their special 2 slices for 4.99 but they did not have any slices so my only choice was to order a whole pizza for just me. I also purchased tiramisu (my absolute favorite dessert that usually no one can mess up) but it was freezer burnt and hard!?!  I usually always enjoy trying mom and pop pizza places at places I visit on travel but this one is definitely one I would not go to again!  I should've driven the extra 10minutes and found a better spot!  Lesson learned!", "labels": [("pizza", "FOOD#QUALITY", "negative"), ("tiramisu", "FOOD#QUALITY", "negative"), ("this one", "RESTAURANT#GENERAL", "negative")]},
    {"text": "Food is not bad (i.e., standard Cosi glorified fast food).  However this particular restaurant has the rudest employees I have ever met in my life.  Asking for a fork is a huge burden on them for which I must apologize.  I am SO sorry I need a fork to eat a salad.  Also the workflow here is weird.  They act like you're a huge idiot for not knowing you order salad at one end and soup at the registers.  Signs might help.  Tomato basil was dec though.", "labels": [("Food", "FOOD#QUALITY", "neutral"), ("employees", "SERVICE#GENERAL", "negative"), ("implicit", "SERVICE#GENERAL", "negative"), ("Tomato basil", "FOOD#QUALITY", "positive")]},
    {"text": "I am very disappointed by their service.  I ordered a Stromboli and chicken wings at 6:15pm for pick up.  I got there 6:50pm but my order wasn't ready.  I just stood there.  At 7:10pm, my order was finally ready.  No words for I waited quite while.  Stromboli was ok, but chicken wings were terrible.  They were cold and over cooked.  They tasted like there days old chicken wings.  Also, they didn't put celery sticks.  I would not recommend this place and of cause, I won't come back!", "labels": [("service", "SERVICE#GENERAL", "negative"), ("Stromboli", "FOOD#QUALITY", "neutral"), ("chicken wings", "FOOD#QUALITY", "negative"), ("celery sticks", "FOOD#STYLE_OPTIONS", "negative"), ("place", "RESTAURANT#GENERAL", "negative")]},
    {"text": "5 stars for the food!  We had the veggie appetizer which was great, and then we each had a pasta dish that we shared. Beet and goat cheese plin, pheasant lasagnette, mushroom ravioli and the tagliatelle with rabbit ragu. Superb (luckily we didn't mind sharing!). Also had sides of the cauliflower rissoto which was delicious and the baked artichokes which was okay.  Bombilinis for dessert was a perfect ending. The wait staff was attentive, although our particular waitress seemed a bit distracted. Am definitely going again to try the pizza next time.", "labels": [("food", "FOOD#QUALITY", "positive"), ("veggie appetizer", "FOOD#QUALITY", "positive"), ("pasta dish", "FOOD#QUALITY", "positive"), ("Bombilinis", "FOOD#QUALITY", "positive"), ("cauliflower rissoto", "FOOD#QUALITY", "positive"), ("baked artichokes", "FOOD#QUALITY", "neutral"), ("wait staff", "SERVICE#GENERAL", "positive"), ("waitress", "SERVICE#GENERAL", "negative")]},
    {"text": "Best donut shop in STL!!! Family owned, clean, truly great service by people that care and know you! The donuts are amazing. Handmade every morning! It's packed every day. Great breakfast sandwiches too!", "labels": [("donut shop", "RESTAURANT#GENERAL", "positive"), ("clean", "AMBIENCE#GENERAL", "positive"), ("service", "SERVICE#GENERAL", "positive"), ("donuts", "FOOD#QUALITY", "positive"), ("breakfast sandwiches", "FOOD#QUALITY", "positive")]},
    {"text": "This place honestly was a disappointment! Very thin cuts of meat for the price compared to other places. The food came out cold too. Had the Dolsot Bibimbap and the stone pot was cold to the point I could've picked it up and not get burned.   When you order a dolsot bibimbap of any kind, the rice at the bottle should be able to get crunchy....nope not here.  LADIES AND GENTLEMEN, please save your money and go somewhere else.", "labels": [("place", "RESTAURANT#GENERAL", "negative"), ("thin cuts of meat", "FOOD#PRICES", "negative"), ("price", "FOOD#PRICES", "negative"), ("food", "FOOD#QUALITY", "negative"), ("stone pot", "FOOD#QUALITY", "negative"), ("dolsot bibimbap", "FOOD#STYLE_OPTIONS", "negative")]}
]

html_rows = []
for r in reviews_data:
    num_labels = len(r['labels'])
    for i, (term, cat, sent) in enumerate(r['labels']):
        if i == 0:
            html_rows.append(f"""
            <tr style="page-break-after: avoid; page-break-inside: avoid;">
                <td rowspan="{num_labels}" style="vertical-align: top; border: 1px solid #ccc; padding: 10px;">{r['text']}</td>
                <td style="border: 1px solid #ccc; padding: 8px;">{term}</td>
                <td style="border: 1px solid #ccc; padding: 8px;">{cat}</td>
                <td style="border: 1px solid #ccc; padding: 8px;">{sent}</td>
            </tr>
            """)
        else:
            html_rows.append(f"""
            <tr style="page-break-inside: avoid;">
                <td style="border: 1px solid #ccc; padding: 8px;">{term}</td>
                <td style="border: 1px solid #ccc; padding: 8px;">{cat}</td>
                <td style="border: 1px solid #ccc; padding: 8px;">{sent}</td>
            </tr>
            """)

final_html_v2 = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        @page {{ size: A4; margin: 15mm; }}
        table {{ width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 9pt; }}
        th {{ background-color: #f2f2f2; border: 1px solid #ccc; padding: 10px; text-align: left; }}
    </style>
</head>
<body>
    <table>
        <thead>
            <tr>
                <th>Review Text</th>
                <th>Aspect Term</th>
                <th>Category</th>
                <th>Sentiment</th>
            </tr>
        </thead>
        <tbody>
            {"".join(html_rows)}
        </tbody>
    </table>
</body>
</html>
"""

with open("detailed_review_analysis_v2.html", "w") as f:
    f.write(final_html_v2)

from weasyprint import HTML
HTML("detailed_review_analysis.html").write_pdf("detailed_review_analysis_v2.pdf")