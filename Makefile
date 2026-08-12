TEMPLATE_JSON_FILES := $(wildcard template_mapping_*.json)

all: csv_created

csv_created : tables.xlsx
	python3 -W ignore tools/xlsx2csv.py -n Category -d ',' -f %Y-%m-%d -q all tables.xlsx Category.csv
	tail -n +2 Category.csv | while IFS= read -r line; do \
                name=$$(echo "$${line}" | awk -F, '{print $$2}' | sed 's#"##g'); \
                python3 -W ignore tools/xlsx2csv.py -n $${name} -d ',' -f %Y-%m-%d -q all tables.xlsx $${name}.csv; \
        done
	touch csv_created

clean:
	rm -f csv_created \
              *.csv \
              *~ \
              *.log
