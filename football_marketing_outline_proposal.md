# Football Marketing Analytics Outline Proposal

## Project Title

**How Football Clubs Reveal Market Preferences Through Transfer Spending**

## Core Argument

Football transfer fees can be read as the **Price** variable in the 4P framework. A club's willingness to pay reveals how it values player positioning, not only player ability.

The main model is **STP**:

- **Segmentation:** classify the player market into meaningful groups.
- **Targeting:** infer which types of players clubs prefer.
- **Positioning:** explain why certain player identities command premiums.

The three core premiums:

- **Growth Premium:** young, high-potential players are priced as future-growth assets.
- **Brand Premium:** league identity works as a brand label and quality signal.
- **Attention Premium:** entertaining attacking players are priced as attention-generating assets.

---

## Slide 1. Opening: Why Football Transfer Price Is a Marketing Question

- **Objective:** Reframe transfer pricing from a sports topic into a marketing analytics topic.
- **Chart to use:** `chart_02_price_skewness.png`
- **Marketing framework:** 4P Price; revealed willingness to pay.
- **Key message:** Transfer fees reveal what clubs value in the market.
- **Speaker note:** Start from the highly skewed price distribution. This shows that football transfer pricing resembles a premium star market, not a normal uniform labor market.

---

## Slide 2. Course Framework Mapping: 5C → STP → 4P Price

- **Objective:** Show how the project fits marketing course concepts.
- **Chart to use:** `chart_01_data_understanding_dashboard.png`
- **Marketing framework:** 5C, STP, 4P.
- **Key message:** Clubs, players, leagues and fans create a structured market where price reflects positioning.
- **Speaker note:** Keep this slide short. Use the data dashboard only to show that the dataset has enough variables for segmentation, targeting and positioning.

---

## Slide 3. Method: Structured Database + Multivariate Marketing Analytics

- **Objective:** Explain the analysis design without overloading the audience.
- **Chart to use:** optional; use `chart_01` as a small supporting visual.
- **Marketing framework:** Market research and segmentation readiness.
- **Key message:** The analysis converts raw football attributes into marketing-facing signals.
- **Speaker note:** Mention engineered features: age group, role group, top-league label, log price, potential gap, attention score and brand score.

---

## Slide 4. Growth Positioning: Youth and Future Value

- **Objective:** Show that young high-potential players are positioned as future assets.
- **Chart to use:** `chart_03_youth_growth_positioning.png`
- **Backup chart:** `chart_13_age_group_log_fee_violin.png`
- **Marketing framework:** STP Targeting; product lifecycle; growth premium.
- **Key message:** The market prices future narratives, not just current performance.
- **Speaker note:** Explain that clubs are buying future upside, resale optionality and brand-building potential. Use young stars as intuitive examples.

---

## Slide 5. Brand Positioning: League Identity and Brand Equity

- **Objective:** Show that league identity functions as market brand equity.
- **Chart to use:** `chart_04_league_brand_ladder.png`
- **Backup chart:** `chart_14_league_role_fee_heatmap.png`
- **Marketing framework:** Brand equity; country-of-origin effect; STP positioning.
- **Key message:** League identity acts like a quality label that frames willingness to pay.
- **Speaker note:** Translate league differences into brand language: quality assurance, visibility, lower perceived risk and higher resale confidence.

---

## Slide 6. Attention Positioning: Entertainment and Visibility Premium

- **Objective:** Show that entertainment-oriented players receive attention premiums.
- **Chart to use:** `chart_05_attention_positioning.png`
- **Marketing framework:** Attention economy; 4P Product and Promotion; attention premium.
- **Key message:** Attacking spectacle creates commercial visibility, not only sporting value.
- **Speaker note:** Explain that pace, dribbling, shooting and reputation make players easier to promote, remember and monetize.

---

## Slide 7. Player Archetypes: From Attributes to Market Identity

- **Objective:** Move beyond descriptive statistics into multivariate positioning.
- **Chart to use:** `chart_06_pca_price_gradient.png`
- **Backup chart:** `chart_07_pca_position_league.png`
- **Marketing framework:** Perceptual mapping; differentiated products.
- **Key message:** Clubs buy recognizable player archetypes, not isolated attributes.
- **Speaker note:** Present PCA as a marketing perceptual map. Avoid technical explanation beyond saying it compresses many attributes into a readable positioning space.

---

## Slide 8. Identity Neighborhoods and Consideration Sets

- **Objective:** Show that players compete within identity-based neighborhoods.
- **Chart to use:** `chart_08_tsne_identity_maps.png`
- **Marketing framework:** Consideration sets; differentiated competition.
- **Key message:** The market compares similar player identities, then prices selected identities at a premium.
- **Speaker note:** Use this to lightly introduce Hotelling-style differentiation: similar players can receive different valuations because they occupy different identity locations.

---

## Slide 9. Segmentation: Market Segments and Targeting Logic

- **Objective:** Name the player groups as market segments.
- **Chart to use:** `chart_10_segmentation_market_segments.png`
- **Backup chart:** `chart_15_segment_archetype_radar.png`
- **Marketing framework:** STP Segmentation and Targeting.
- **Key message:** Different player segments sell different value propositions.
- **Speaker note:** Explain segments as product bundles: growth assets, creative connectors, utility players. This makes clustering useful for marketing rather than just statistics.

---

## Slide 10. Price Model: Transfer Fee as Revealed Willingness to Pay

- **Objective:** Convert the narrative into premium interpretation.
- **Chart to use:** `chart_11_hedonic_price_premiums.png`
- **Marketing framework:** 4P Price; hedonic pricing.
- **Key message:** Clubs pay implicit premiums for quality, growth, brand and attention.
- **Speaker note:** Do not present this as an econometrics result. Present it as a clean decomposition of revealed willingness to pay.

---

## Slide 11. Integrated STP Positioning Map

- **Objective:** Provide a consulting-style synthesis slide.
- **Chart to use:** `chart_16_integrated_stp_positioning_map.png`
- **Marketing framework:** STP Positioning.
- **Key message:** Premium players often combine future potential and attention value.
- **Speaker note:** This is the main strategic map. Use it to summarize the three positioning dimensions before conclusion.

---

## Slide 12. Light Competition Intuition: Differentiated Player Markets

- **Objective:** Add competition intuition without making economics the main story.
- **Chart to use:** `chart_12_transfer_premium_heatmap.png`
- **Optional appendix chart:** `chart_17_league_transfer_flow_heatmap.png`
- **Marketing framework:** Revealed preference; light differentiated competition.
- **Key message:** Actual payments and transfer flows support the positioning story.
- **Speaker note:** Keep this secondary. Avoid making scarcity premium the core claim. Mention that clubs compete for differentiated identities, not generic players.

---

## Slide 13. Integrated Conclusion

- **Objective:** Close with the central marketing answer.
- **Chart to use:** `chart_09_positioning_maps.png`
- **Marketing framework:** STP + 4P Price.
- **Key message:** Transfer fee is the price of positioning under uncertainty.
- **Speaker note:** Summarize the three premiums: growth, brand and attention. End with: clubs do not simply buy football ability; they buy differentiated market positions.

---

## Backup / Appendix Charts

Use these if the audience asks for more detail:

- `chart_01`: data credibility and segmentation readiness.
- `chart_07`: PCA by position and league.
- `chart_13`: age-group distribution detail.
- `chart_14`: league × role price heatmap.
- `chart_15`: segment archetype radar.
- `chart_17`: static transfer-flow heatmap.

Do **not** use scarcity premium as a core slide. If an older scarcity-premium chart exists, place it in appendix or exclude it from the main story because it shifts the project toward economics / auction theory. The main story should remain marketing-first: **Growth Premium, Brand Premium, Attention Premium**.

---

## Recommended 10-Minute Flow

1. 0:00-1:00 Opening and research question.
2. 1:00-2:00 Framework: STP + 4P Price.
3. 2:00-3:10 Growth Positioning.
4. 3:10-4:20 Brand Positioning.
5. 4:20-5:30 Attention Positioning.
6. 5:30-6:40 PCA / t-SNE identity maps.
7. 6:40-7:40 Segmentation.
8. 7:40-8:40 Price premium decomposition.
9. 8:40-9:30 Integrated STP map.
10. 9:30-10:00 Final conclusion.

## Final One-Sentence Argument

**Football clubs do not simply buy player ability; they buy growth narratives, brand assurance and attention value, and transfer fee is the market price of that positioning.**
