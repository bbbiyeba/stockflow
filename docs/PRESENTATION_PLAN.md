# Presentation plan: Stockflow

This plan covers the deck's two projects, the **demand forecast** and the **rebalancing optimizer**, plus a live demo of the dashboard at [stockflow-six-mu.vercel.app](https://stockflow-six-mu.vercel.app).

**Format:** 15 minutes in total.
- 8 minutes of slides.
- 3 minutes of live demo.
- 4 minutes of questions.

**Materials:**
- [Google Slides deck](https://docs.google.com/presentation/d/1yX3KXnzmiQCkrfQPBK_SOoII-ivtylqnIXNmuuVUWxA/edit) (speaker notes included).
- `presentation/Stockflow_Stryker.pptx` and the PDF as backup.
- The hosted dashboard.

All numbers come from the v1.0.1 run on **synthetic data**. Say so once, on slide 1.

## Storyline: a story in five beats

The talk is told as one story: **a surgeon is waiting for a part that is sitting in the wrong warehouse.** Each slide moves that story forward rather than just presenting results.

| Beat | Slides | The story says | Feeling to leave |
|---|---|---|---|
| 1. The hook | 1 | A patient is waiting for a part that exists, just at the wrong branch | "That shouldn't happen" |
| 2. The villain | 2 | It isn't a shortage. We have *too much* stock and *too little* at once | Surprise |
| 3. The quest | 3–4 | First I had to see the future: forecast demand at every branch. Then I hit a setback (the bias) and fixed it | Trust |
| 4. The turn | 5–7 | With a forecast I trust, I can move stock *before* anyone runs out, and a manager stays in control | "This could work here" |
| 5. The ending | 8 | The patient doesn't wait. Here is how we make that real: a 12-week pilot | "Let's do it" |

**The thread to keep coming back to:** the waiting patient. Mention them at the start (slide 1), in the middle (slide 5), and in the last line (slide 8). That callback is what makes the talk feel like a story rather than a report.

**Connecting lines between slides** (say these word for word; they carry the flow):
- 1 → 2: "So why does that happen?"
- 2 → 3: "To fix placement, I first needed to know where demand will be."
- 3 → 4: "Everything rests on the forecast, so let's start there."
- 4 → 5: "A forecast alone doesn't help the patient. Moving the stock does."
- 5 → 6: "So what does that add up to?"
- 6 → 7: "Numbers are one thing. Let me show you what a branch manager actually sees."
- 7 → 8: "So where do we go from here?"

## Slide scripts

### Slide 1: The hook (0:45)

> "Picture a surgeon at 7 a.m., ready to operate, and the implant kit they need isn't at the local branch. It's sitting on a shelf at another branch three states away, where nobody needs it this month. The patient waits. The frustrating part is that the product *existed*. It was just in the wrong place.
>
> I'm Bryce Biyeba, and this summer I built Stockflow to stop that from happening. It does two things: it predicts what each branch will need, and it moves stock there before anyone runs out. One note before we start: I built it on synthetic data designed to behave like our real ERP extracts. So the method is real, and the dollar figures are illustrative."

- **Point at:** the map. "25 branches, 300 products, almost 4,900 combinations to get right."
- **Then:** "So why does that happen?"

### Slide 2: The villain (1:00)

> "When I looked at the inventory, I expected to find that we were short. We weren't. 62% of our inventory dollars sit *above* target. That's $8.3M of the $13.4M on hand. Yet at the same moment, 419 branch positions are about to run out.
>
> So the villain isn't a shortage. It's placement. We have too much and too little at the same time, just in different places."

- **Point at:** the long black Overstocked bar ($7.3M), then the tiny gold At-risk bar (419 positions). "Most of the money is here, and all of the pain is here."
- **Aside:** "$3.7M hasn't sold anywhere in 90 days. Moving it won't help. That's a decision for liquidation, and the tool says so honestly."
- **Then:** "To fix placement, I first needed to know where demand will be."

### Slide 3: The plan (0:45)

> "So I built one pipeline that goes from raw data to a decision. Three years of sales, stock moves and purchase orders come in, and every input is checked on the way. Then I forecast demand, set an inventory target for each branch, and an optimizer works out which transfers to make. The dashboard explains each one in plain English.
>
> And the last box matters most. Nothing moves until a planner says yes. This tool advises; people decide."

- **Point at:** the gold "Planner approval" box.
- **Then:** "Everything rests on the forecast, so let's start there."

### Slide 4: The quest and the setback (1:45)

> "To be useful, the forecast had to beat what a planner would do by hand. I tested it the honest way: I trained it on history, then scored it on a month it had never seen. It's 33% more accurate than the seasonal baseline: 49.8% daily error against 74.9%.
>
> But here's the part I'm proudest of, because it went wrong first. When I scaled the network to 250 branches, I found the forecast was quietly running about 2.6% *low*, which means ordering too little every time. My first theory about the cause was wrong; I tested it and it didn't hold. The real cause was that the model under-shot certain kinds of products. So I built a correction for each product segment that learns only from past months. Bias dropped to zero here and to −0.8% at 10× scale, with no loss of accuracy."

- **Point at:** the chart. "Black is what actually sold, gold is the forecast, and the shaded band is the likely range. Real demand landed inside it 83% of the time; the target was 80%."
- **If asked about the 50% daily error:** "Single days for slow movers are mostly noise. Over the four weeks we actually plan on, error is 12.8%."
- **Then:** "A forecast alone doesn't help the patient. Moving the stock does."

### Slide 5: The turn (1:15)

> "Remember that surgeon? This is where they get their implant. The optimizer takes every forecast and every branch's target, and asks one question: which moves save the most, after paying freight? It respects truck capacity and minimum shipments, and it never leaves a sending branch short itself.
>
> The answer is 895 transfers. They cost $46k to ship and save $216k net. That's $11k more than a sensible rule of thumb."

- **Point at:** the map ("black branches send, gold branches receive"), then one table row.
- **Say:** "And every move comes with a reason a branch manager can check, like 'excess here covers a shortfall there'."
- **Then:** "So what does that add up to?"

### Slide 6: The payoff (1:00)

> "In a single planning cycle, $1.2M of idle stock goes back to work, and we avoid 85% of expected lost sales: 8,948 units that would otherwise have been a customer or a patient told 'not today'. Positions at risk drop from 419 to 76. Inventory turns go up from 8.5 to 9.0."

- **Point at:** the gold $1.2M box, then the before/after column.
- **Honesty line:** "The cost of a lost sale is an assumption finance would need to agree. Even if I change one assumption at a time, savings stay between $182k and $248k."
- **Then:** "Numbers are one thing. Let me show you what a branch manager actually sees."

### Slide 7: Into the product (0:30), then the live demo

> "Everything comes down to one screen. You get the plan in one sentence, a map of where stock flows, and an approve or reject button on every move. Let me show you live."

- **Action:** switch to the browser tab you opened beforehand.

### Slide 8: The ending (1:00)

> "So, how do we make this real? I'd propose a 12-week pilot. In the first four weeks, we connect real data from two or three branches. For the next six weeks, it runs in shadow mode. Planners see every recommendation and approve or reject it, but nothing changes automatically. In the last two weeks, we compare against what really happened and decide.
>
> It has limits. It's been on synthetic data so far, it plans one cycle at a time, and the costs need finance's sign-off. That's exactly what the pilot is for.
>
> Back to that surgeon at 7 a.m. With this, the implant is already at the branch. Right product, right branch: surgeons don't wait, so patients don't either. Thank you."

## Live demo: stockflow-six-mu.vercel.app (3:00)

**Before the talk (T−10 min):**
1. Open https://stockflow-six-mu.vercel.app in a browser window 1440 px wide, and let it load fully. The first load of a cold container can take 20–40 s.
2. Click through every page once to warm the caches.
3. End on **Transfer recommendations**.
4. Keep the PDF open as the fallback. Slide 7 has the annotated screenshot if the network fails.

**The hosted demo is view-only.** Approve/Reject is shown but not saved. Say: "Locally, every decision is written to an audit log."

| Time | Page | Do | Say |
|---|---|---|---|
| 0:00 | Transfer recommendations | Point at the kicker line | "Same numbers as the slides, live. The line at the top: nothing moves until a planner approves." |
| 0:15 | Headline metrics | Point at **$216k** and **8,948** | "895 transfers, $46k freight, $216k net savings, 8,948 lost-sale units covered." |
| 0:30 | Expand **Before / after** | Run a finger down the table | "Fill rate goes from 93.5% to 99.0%, and at-risk positions from 419 to 76. Every formula is written out here, nothing is a black box." |
| 0:50 | Corridor map → transfer table | Hover a corridor; point at a **Reason** cell and the Approve column | "Each row is one move, with its reason. A planner approves or rejects, and that's logged with name, time and run." |
| 1:15 | **How sensitive is the plan?** | Point at the savings range | "Change one assumption at a time: savings stay between $182k and $248k, and most moves survive." |
| 1:30 | **Inventory health** | Point at the darkest heatmap cells | "This is where risk concentrates by branch and category. The $3.7M of stagnant stock is flagged for liquidation, not shuffled around." |
| 1:55 | **Forecast explorer** (B01, top SKU) | Point at the band, the red stock-out days, then the drivers | "This is project one up close: history, the next 28 days with a range, stock-out days excluded because sales understate demand there, and the drivers in plain language." |
| 2:25 | **Forecast accuracy** | Point at the headline, then **Where the model does not win** | "33% better than baseline, zero bias. It beats the best simple model in every branch, category and class. It still loses on 19% of individual series, mostly slow movers, and we show that rather than hide it." |
| 2:50 | Back to the slides (slide 8) | — | "So that's the tool. Where do we go from here?" |

**If something breaks:**
- **Page is slow:** keep talking through the slide 7 screenshot.
- **Site is down:** say "the network's not cooperating" and walk slide 7's callouts 1–3 instead. Don't debug live.

## Q&A prep

| Question | Answer |
|---|---|
| Can it raise transfers in the ERP? | Deliberately not. It recommends; a person decides. |
| Why LightGBM and not a deep model? | One model across 4,900 series trains in minutes and is explainable with SHAP. It already beats every baseline in every segment. |
| How do you know it's not overfit? | It is scored on a rolling-origin backtest over 4 folds. Model choice uses folds 1–3; the reported fold 4 is untouched. A leakage test guards it in CI. |
| What about the 19% of series where it loses? | Mostly slow, intermittent items. The selector picks per segment, and safety stock doesn't depend on the point forecast alone. |
| What does it cost to run? | About 6–8 minutes on one 4-core machine, all open source. |
| What data would you need from us? | Seven standard extracts. The schemas are in the README, under "Connecting real data". |
| How does it scale? | Tested at 10× (250 branches): still 33% better than baseline, bias −0.8%. The optimizer solves in parallel SKU batches. |
| What's the biggest risk? | Real data quality, which is why the pilot runs in shadow mode first. |

## Rehearsal checklist

- [ ] Run through the slides twice with a timer. Target 8:00; cut the slide 4 scale point if over.
- [ ] Run the demo twice on the hosted site, including the "site is down" fallback.
- [ ] Memorise the four numbers: **33%** better forecast, **$216k** net, **$1.2M** freed, **419 → 76** at risk.
- [ ] Say "synthetic data" exactly once, on slide 1.
- [ ] Hit the surgeon callback three times: slides 1, 5 and 8.
- [ ] Practise the connecting lines until they come out without notes.
