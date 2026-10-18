# Demo script: 3 minutes

**Setup:**
- Run `make app` (or open the hosted view-only demo at stockflow-six-mu.vercel.app) and start on **Transfer recommendations**.
- Use a 1440-pixel-wide window. Collapse nothing.
- Every number below is from the current pipeline run, on **synthetic data**. Say so once, at the start.

| Time | Screen | Do | Say |
|---|---|---|---|
| 0:00 | Transfer recommendations, top | Point at the kicker line | "This is Stockflow. It forecasts demand for every product at every branch, and recommends moving stock between branches before anyone runs out. The data is synthetic, built to behave like a real distributor's ERP extracts, so the method is real and the dollar figures are illustrative. Note the line at the top: nothing moves until a planner approves it." |
| 0:20 | Headline and the four metrics | Point at **$216k** and **8,948** | "The plan in one sentence: 895 transfers of 35,693 units cost $46k in freight and save $216k net. They cover 8,948 units of expected lost sales and free $1.2M of excess. That is $11k more than a simple rule of thumb." |
| 0:40 | Expand **Before / after, and how these numbers are calculated** | Run a finger down the table | "Availability goes from a 93.5% to a 99.0% expected fill rate, and positions at stock-out risk drop from 419 to 76. Turns go from 8.5 to 9.0. Every formula and assumption is written out here. The cost of a lost sale is an estimate finance would need to agree." |
| 1:00 | Scroll to the corridor map, then the transfer table | Hover one corridor; point at a **Reason** cell | "The map shows where stock flows. Each row is one move with a plain-language reason, for example excess at one branch covering a shortfall at another. A planner sets Approve or Reject, and the decision is logged with their name, the time and the run." |
| 1:25 | Scroll to **How sensitive is the plan?** | Point at the savings range | "If I change one assumption at a time (service level, freight cost, forecast error or forecast level), most of the same moves survive and savings stay between $182k and $248k. The plan doesn't hinge on one guess." |
| 1:40 | **Inventory health** | Point at the heatmap's darkest cells | "Here is why we need it. 419 positions are likely to stock out, while $8.3M sits above target. The grid shows which branch and category pairs are at risk. $3.7M is stagnant stock nobody in the network will buy; that is a liquidation call, and the tool says so rather than shuffling it around." |
| 2:05 | **Forecast explorer** (B01, top SKU) | Point at the band, then the drivers list | "For any branch and product: the history, the next 28 days with a 10th–90th percentile range, and the stock-out days in red, which we exclude because sales understate demand on those days. Below it, the main drivers in plain language: here, the same weekday over the last four weeks." |
| 2:25 | **Forecast accuracy** | Point at the headline, then **Where the model does not win** | "On a month the model never saw, error is 49.8% of daily volume, against 74.9% for the seasonal baseline: 33% better, with no overall bias: each segment's forecast is calibrated on earlier months only. It beats the best simple model in every branch, category and class. Series by series it still loses on 19%, mostly slow movers, and that is shown rather than hidden." |
| 2:45 | Back to **Transfer recommendations** | — | "Next step: a 12-week shadow pilot at two or three branches on real extracts, with planners approving or rejecting each move. Getting the right product to the right branch is one concrete way we make healthcare better, together with our customers. Happy to take questions." |

**If asked:**
- **"Can it raise the transfer in the ERP?"** Deliberately not. It recommends; a person decides.
- **"Is daily error of 50% bad?"** Single days of slow movers are mostly noise. Planning uses 28-day and lead-time totals, where error is 12.8%.
- **"What does it cost to run?"** About 8 minutes on one 4-core machine. See [FAQ.md](FAQ.md).
