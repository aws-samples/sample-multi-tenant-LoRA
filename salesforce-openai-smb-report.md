# Salesforce Report: OpenAI/Codex Opportunities in SMB Territories (2026)

Scope: All 2026 opportunities in territories prefixed with `NAMED-AGS-NAMER-UNITED STATES-SSI-SMB` that mention OpenAI, Open AI, or Codex in name, description, next step, competitor, or details fields.

---

## Option 1: SOQL Query (Developer Console / Workbench / Data Loader)

Paste this into Workbench (workbench.developerforce.com) → Queries → SOQL Query, or Developer Console → Query Editor.

```sql
SELECT
    Id,
    Name,
    Account.Name,
    Account.Territory_Lookup__r.Name,
    Owner.Name,
    Owner.Alias,
    StageName,
    Amount,
    CloseDate,
    RecordType.Name,
    Primary_Competitor__c,
    Details_if_Other_or_No_Competition__c,
    Primary_Partner_Name__c,
    Co_Sell_Engagement_Type__c,
    LeadSource,
    NextStep,
    Description,
    LastModifiedDate,
    CreatedDate
FROM Opportunity
WHERE
    Account.Territory_Lookup__r.Name LIKE 'NAMED-AGS-NAMER-UNITED STATES-SSI-SMB%'
    AND CloseDate >= 2026-01-01
    AND CloseDate <= 2026-12-31
    AND (
        Name LIKE '%OpenAI%'
        OR Name LIKE '%Open AI%'
        OR Name LIKE '%Codex%'
        OR Description LIKE '%OpenAI%'
        OR Description LIKE '%Open AI%'
        OR Description LIKE '%Codex%'
        OR NextStep LIKE '%OpenAI%'
        OR NextStep LIKE '%Open AI%'
        OR NextStep LIKE '%Codex%'
        OR Primary_Competitor__c = 'OpenAI'
        OR Details_if_Other_or_No_Competition__c LIKE '%OpenAI%'
        OR Details_if_Other_or_No_Competition__c LIKE '%Codex%'
        OR Primary_Partner_Name__c = 'OpenAI'
    )
ORDER BY StageName, Amount DESC
```

### Aggregate variant (counts and $ by stage)

```sql
SELECT
    StageName,
    COUNT(Id) opp_count,
    SUM(Amount) total_amount
FROM Opportunity
WHERE
    Account.Territory_Lookup__r.Name LIKE 'NAMED-AGS-NAMER-UNITED STATES-SSI-SMB%'
    AND CloseDate >= 2026-01-01
    AND CloseDate <= 2026-12-31
    AND (
        Name LIKE '%OpenAI%'
        OR Name LIKE '%Codex%'
        OR Description LIKE '%OpenAI%'
        OR Description LIKE '%Codex%'
        OR NextStep LIKE '%OpenAI%'
        OR NextStep LIKE '%Codex%'
        OR Primary_Competitor__c = 'OpenAI'
        OR Primary_Partner_Name__c = 'OpenAI'
    )
GROUP BY StageName
ORDER BY StageName
```

---

## Option 2: Salesforce Lightning Report (visual builder)

### Report Type
**Opportunities with Account** (standard) — this ensures `Account.Territory_Lookup__r` fields are available for filtering.

### Filters

| # | Field | Operator | Value |
|---|---|---|---|
| 1 | Account: Territory Name (`Territory_Lookup__r.Name`) | starts with | `NAMED-AGS-NAMER-UNITED STATES-SSI-SMB` |
| 2 | Close Date | greater or equal | `1/1/2026` |
| 3 | Close Date | less or equal | `12/31/2026` |
| 4 | Opportunity Name | contains | `OpenAI` |
| 5 | Opportunity Name | contains | `Codex` |
| 6 | Description | contains | `OpenAI` |
| 7 | Description | contains | `Codex` |
| 8 | Next Step | contains | `OpenAI` |
| 9 | Primary Competitor | equals | `OpenAI` |
| 10 | Primary Partner Name | equals | `OpenAI` |
| 11 | Details if Other or No Competition | contains | `OpenAI` |

### Filter Logic (paste into Filter Logic field)

```
1 AND 2 AND 3 AND (4 OR 5 OR 6 OR 7 OR 8 OR 9 OR 10 OR 11)
```

### Grouping
- Group by **Stage** (rows)
- Summarize **Amount** (Sum) at the group level
- Add **Record Count** in summary

### Columns to display
- Opportunity Name
- Account Name
- Territory Name
- Owner Full Name
- Stage
- Amount
- Close Date
- Primary Competitor
- Primary Partner Name
- Last Modified Date

### Chart (optional)
Stacked bar: Amount by Stage, colored by Owner or Territory.

---

## Option 3: SF CLI (if you have `sf` installed and an authenticated org)

```bash
sf data query --query "$(cat <<'EOF'
SELECT Id, Name, Account.Name, Account.Territory_Lookup__r.Name,
       Owner.Alias, StageName, Amount, CloseDate,
       Primary_Competitor__c, Primary_Partner_Name__c
FROM Opportunity
WHERE Account.Territory_Lookup__r.Name LIKE 'NAMED-AGS-NAMER-UNITED STATES-SSI-SMB%'
  AND CloseDate >= 2026-01-01 AND CloseDate <= 2026-12-31
  AND (Name LIKE '%OpenAI%' OR Name LIKE '%Codex%'
       OR Description LIKE '%OpenAI%' OR NextStep LIKE '%OpenAI%'
       OR Primary_Competitor__c = 'OpenAI' OR Primary_Partner_Name__c = 'OpenAI')
ORDER BY StageName, Amount DESC
EOF
)" --result-format csv --target-org aws-crm > openai_smb_2026.csv
```

---

## Option 4: SOSL (better for long-text field matching)

SOQL `LIKE` on Long Text fields only scans the first ~4000 characters. If sellers put OpenAI/Codex deep in Description, use SOSL instead.

```
FIND {OpenAI OR Codex OR "Open AI"} IN ALL FIELDS
RETURNING Opportunity(
    Id, Name, Account.Name, Account.Territory_Lookup__r.Name,
    Owner.Alias, StageName, Amount, CloseDate,
    Primary_Competitor__c, Primary_Partner_Name__c
    WHERE Account.Territory_Lookup__r.Name LIKE 'NAMED-AGS-NAMER-UNITED STATES-SSI-SMB%'
      AND CloseDate >= 2026-01-01 AND CloseDate <= 2026-12-31
    ORDER BY StageName, Amount DESC
)
```

---

## Notes and gotchas

1. **Active vs inactive territories**: Some territories in the SMB prefix are marked `active__c = false`. Add `AND Account.Territory_Lookup__r.Active__c = TRUE` to restrict to currently active territories.
2. **`Description` and other long text fields**: SOQL `LIKE` only supports the first ~4,000 chars. Use SOSL (Option 4) for deep matches.
3. **Task/Note fields**: If sellers document OpenAI mentions in Task or ContentNote records instead of the opp fields themselves, add a separate SOSL query against Task and ContentNote.
4. **Custom fields**: `Primary_Competitor__c`, `Primary_Partner_Name__c`, `Details_if_Other_or_No_Competition__c`, etc. are AWS Salesforce-specific — they exist in aws-crm.lightning.force.com but not in a vanilla SFDC org.
5. **Row limit**: SOQL UI caps at 50,000 rows; use the aggregate query first to size the result set.
6. **Field-level security**: You'll only see records/fields you have permission to view — enterprise-wide runs should be executed by a user with a broad-visibility profile (e.g., sales ops or field ops).

---

## Sanity-check row-count query

```sql
SELECT COUNT() FROM Opportunity
WHERE Account.Territory_Lookup__r.Name LIKE 'NAMED-AGS-NAMER-UNITED STATES-SSI-SMB%'
  AND CloseDate >= 2026-01-01 AND CloseDate <= 2026-12-31
  AND (Name LIKE '%OpenAI%' OR Name LIKE '%Codex%'
       OR Primary_Competitor__c = 'OpenAI' OR Primary_Partner_Name__c = 'OpenAI')
```

Run this first to confirm the filter is scoped correctly before pulling the full detail query.
