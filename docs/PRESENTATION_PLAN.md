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

## Storyline

The talk answers one question: **why do we have too much stock and too little at the same time, and what do we do about it?**

| Part | Slides | Project | Message | Time |
|---|---|---|---|---|
| Problem | 1–2 | — | Stock is in the wrong place, not short overall | 1:45 |
| How | 3 | Both | One pipeline, with a human approval at the end | 0:45 |
| Project 1 | 4 | Forecast | We can predict demand per branch and product, without bias | 1:45 |
| Project 2 | 5–6 | Optimizer | Turning the forecast into transfers that pay for themselves | 2:15 |
| Product | 7 | Both | A manager uses it on one screen → **switch to the live demo** | 0:30 + 3:00 demo |
| Ask | 8 | — | A 12-week pilot | 1:00 |

**Transitions to say out loud** (they hold the two projects together):
- From 2 to 3: "To fix placement, we need to know where demand will be, then decide what to move."
- From 4 to 5: "A forecast alone changes nothing. Here is what it lets us do."
- From 6 to 7: "Let me show you what a branch manager actually sees."

## Slide scripts

### Slide 1: Stockflow (0:30)

> "I'm Bryce Biyeba. Over the internship I built Stockflow. It is two connected projects. The first forecasts daily demand for every product at every branch. The second uses those forecasts to recommend moving stock between branches before anyone runs out. The aim is the right product at the right branch, so surgeons, hospitals and their patients aren't left waiting. One caveat up front: the data is synthetic, built to behave like real ERP and warehouse extracts. The method is real; the dollar figures are illustrative."

- **Point at:** the map. "25 branches, 300 products, about 4,900 branch-product combinations."

### Slide 2: The problem (1:15)

> "The headline: 62% of inventory dollars sit above target, and at the same time 419 branch positions face stock-outs. $8.3M of the $13.4M on hand is above target, and on 5.8% of branch-product days over three years we had nothing on the shelf. So this isn't a shortage problem. It's a placement problem: the stock exists, it's just at the wrong branch."

- **Point at:** the bar chart. Overstocked is $7.3M; at-risk is only $0.2M but 419 positions.
- **Say:** "$3.7M hasn't sold anywhere in 90+ days. No transfer fixes that. It's a liquidation decision, and the tool flags it as one."

### Slide 3: The pipeline (0:45)

> "One pipeline does this end to end. Standard extracts come in: sales, stock moves, purchase orders, promotions. Every input is validated on load. Then come forecasting (project one), the inventory policy, and the rebalancing optimizer (project two). A dashboard explains each move in plain language. The last box is the important one: nothing moves without a planner's sign-off."

- **Point at:** the gold "Planner approval" box.
- The full run takes about six minutes on one laptop.

### Slide 4: Project 1, the forecast (1:45)

> "Project one. The forecast is a single LightGBM model across all branches and products, compared against four simpler methods. To test it honestly, I trained on history and scored it on a month it never saw. Daily error is 49.8% of volume, against 74.9% for the seasonal baseline: 33% more accurate. Bias is zero. Each segment's forecast is calibrated using only earlier months, so we neither over- nor under-order systematically."

- **Point at:** the chart. "Black is actual, gold is forecast, and the shaded band is the 10th–90th percentile range. It held 83% of actual days against an 80% target."
- **Expect this:** "50% sounds high." Answer: "That's single days, where slow movers are mostly noise. Over the 28-day and lead-time totals we plan on, error is 12.8%."
- **Optional scale point:** at 10× scale (250 branches), it is still 33% better than the baseline, and bias is −0.8% after calibration, from −2.6% before.

### Slide 5: Project 2, the transfers (1:15)

> "Project two takes the forecasts and the inventory targets and solves an optimization. Which transfers, from where to where, maximize net savings? It is subject to truck capacity, minimum shipment sizes, and never pushing a sender below its own safety stock. The answer is 895 transfers. They cost $46k to move and save $216k net, which is $11k more than a sensible rule of thumb would."

- **Point at:** the map ("black sends, gold receives"), then one table row.
- **Say:** "Every row has a reason a branch manager can check, like 'excess to shortfall'."

### Slide 6: Impact (1:00)

> "In one planning cycle the plan frees $1.2M of excess inventory and avoids 8,948 lost-sale units, 85% of the expected lost sales. Positions at stock-out risk fall from 419 to 76, and turns rise from 8.5 to 9.0."

- **Point at:** the gold $1.2M box, then the before/after table.
- **Caveat to say:** "The cost of a lost sale is an assumption finance would need to agree. The dashboard shows that savings stay between $182k and $248k across a sensitivity range."

### Slide 7: The dashboard (0:30), then the live demo

> "All of this lands on one screen: the plan in one sentence, where stock flows, and an approve/reject on each move that is logged. Rather than describe it, let me show you."

- **Action:** switch to the browser tab you opened beforehand (see the demo below).

### Slide 8: Next steps (1:00)

> "The ask is a 12-week pilot. Weeks 1–4: connect the seven standard extracts from two or three branches. Weeks 5–10: run in shadow mode, where planners approve or reject each recommendation but we don't act on it automatically. Weeks 11–12: compare the result against what actually happened and decide. The limitations are on the slide. It's synthetic so far, it plans one replenishment cycle, and costs need finance sign-off. Right product, right branch: surgeons don't wait, so patients don't either. Thank you."

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
| 2:50 | Back to the slides (slide 8) | — | "So that's the tool. Here's what I'd like to do next." |

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
