// "What this scheme pays": the reward_table preview rows as an HTML table.

import { formatNumber, h } from "./dom.js";

export function rewardTable(rows) {
  return h(
    "table",
    {},
    h("thead", {}, h("tr", {}, h("th", { text: "Situation" }), h("th", { class: "num", text: "Reward" }), h("th", { text: "Terms" }))),
    h(
      "tbody",
      {},
      rows.map((row) =>
        h(
          "tr",
          {},
          h("td", { text: row.situation }),
          h("td", { class: "num", text: formatNumber(row.reward, 4) }),
          h(
            "td",
            {},
            Object.keys(row.breakdown).length === 0
              ? h("span", { class: "muted", text: "nothing" })
              : h("div", { class: "chips" }, Object.entries(row.breakdown).map(([term, v]) => h("span", { class: "chip", text: `${term} ${formatNumber(v, 4)}` }))),
          ),
        ),
      ),
    ),
  );
}
