-- Typeset the saved Markdown tables as floats in a two-column article.
-- This filter changes presentation only; cell values are retained.
local table_number = 0
local captions = {
  'Historical regular-session bar coverage. AAPL is the only forecast target.',
  'Registered search sizes per information set and outer fold. Configurations have model-specific computing costs.',
  'Development results, 2023Q1--2025Q2. Gains are relative to the same-panel HAR; negative gains denote higher loss. All gates refers to the joint qualification rule.',
  'Annual-segment relative QLIKE improvements against the matching HAR. Better quarters counts negative daily-loss differences; the seed rule checks pooled and annual directions for the prescribed seeds.',
  'Development empirical coverage and mean width. Widths are on the RV scale, multiplied by $10^5$; quantities use equal session weights.',
  'Daily-mean observed-on-predicted regression. Intercepts use RV units multiplied by $10^5$. Slope standard errors use the saved five-lag HAC covariance; no new joint calibration test is performed.',
  'Historical confirmation, 2025H2--2026H1. Intervals are five-session block intervals for Ridge minus HAR.'
}
local short_headers = {
  ['Information set'] = 'Inputs',
  ['Unique configurations per information set and outer fold'] = 'Configs.',
  ['QLIKE improvement vs. corresponding HAR'] = 'Gain (\\%)',
  ['Five-session 95% difference interval'] = '5-session 95\\% CI',
  ['80% width × 10⁵'] = '80\\% width $\\times10^5$',
  ['90% width × 10⁵'] = '90\\% width $\\times10^5$'
}
short_headers['2023 gain (%)'] = '2023 (\\%)'
short_headers['2024 gain (%)'] = '2024 (\\%)'
short_headers['2025H1 gain (%)'] = '2025H1 (\\%)'
short_headers['Better quarters'] = 'Quarters'
short_headers['Intercept × 10⁵'] = 'Intercept $\\times10^5$'
short_headers['HAC SE(slope)'] = 'HAC SE'
short_headers['Configurations'] = 'Configs.'

local function cell_text(cell, header)
  local plain = pandoc.utils.stringify(cell.contents)
  if header and short_headers[plain] then return short_headers[plain] end
  return pandoc.write(pandoc.Pandoc(cell.contents), 'latex'):gsub('%s+$', '')
end

function Table(el)
  table_number = table_number + 1
  local n = #el.colspecs
  local single = table_number == 1 or table_number == 2
  local environment = single and 'table' or 'table*'
  local spec = string.rep('l', n)
  if table_number == 1 then
    spec = 'lrrrl'
  elseif table_number == 2 then
    spec = 'lp{.12\\columnwidth}p{.42\\columnwidth}'
  elseif table_number == 3 then
    spec = 'llrrrrl'
  elseif table_number == 4 then
    spec = 'llrrrrl'
  elseif table_number == 5 then
    spec = 'llrrrr'
  elseif table_number == 6 then
    spec = 'lllrrr'
  elseif table_number == 7 then
    spec = 'lrrrrl'
  end
  local parts = {'\\begin{'..environment..'}[!t]', '\\centering', '\\footnotesize',
    '\\caption{'..captions[table_number]..'}', '\\label{tab:results-'..table_number..'}',
    '\\begin{tabular}{'..spec..'}', '\\toprule'}
  local function append_rows(rows, header)
    for _, row in ipairs(rows) do
      local cells = {}
      for _, cell in ipairs(row.cells) do
        local value = cell_text(cell, header)
        if table_number == 1 and not header then
          value = value:gsub('Forecast target and historical input', 'Target/input')
          value = value:gsub('Historical market input', 'Market input')
        end
        table.insert(cells, value)
      end
      table.insert(parts, table.concat(cells, ' & ')..' \\\\')
    end
  end
  append_rows(el.head.rows, true)
  table.insert(parts, '\\midrule')
  for _, body in ipairs(el.bodies) do
    append_rows(body.head, false)
    append_rows(body.body, false)
  end
  table.insert(parts, '\\bottomrule')
  table.insert(parts, '\\end{tabular}')
  table.insert(parts, '\\end{'..environment..'}')
  return pandoc.RawBlock('latex', table.concat(parts, '\n'))
end

function Header(el)
  if pandoc.utils.stringify(el.content) == 'References' then
    return pandoc.RawBlock('latex', '\\section*{References}\n\\begingroup\\small')
  end
  el.level = el.level - 1
  return el
end
