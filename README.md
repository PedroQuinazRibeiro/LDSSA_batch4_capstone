# batch4-capstone of Lisbon Data Science Starters Academy

## Summary of client requirements

The goal of this project is to analyze and improve the stop and search policy of
the United Kingdom Police Department, that’s supposed to ensure that officers
stop people and cars when there is probable cause. There are 41 police stations
in the United Kingdom, each with its own policy.  There have been accusations in
the press that the police tend to stop and search certain minorities at a higher
rate than others and that during searches women of certain age groups and
ethnicities are asked to remove cloth more frequently than other groups.

Therefore the first task of the project is to search for evidence that any
police station may be discriminating on gender, ethnicity or age regarding who
they chose to stop, and also who they ask for any clothing to be removed.

If the previous task is successful, the second task of the project is to create
and deploy an application for authorizing searches. This application should
ensure uniformity of stop and search policies.  It will be integrated on the
police system that officers need to use and will be running during 2020. A
search should be authorized only when there is likelihood above 10% for the
search to be successful. This requirement will be evaluated per station and
search objective. Furthermore it is also expected that the application should be
able to level the search success rate between ethnicities for every station and
for every search objective,  without significantly diminishing the current
ability to detect offences.

## Deliverables

The code of the application can be found in folder "app". It has been deployed
to Heroku.

The analysis of data and the model evaluation is discussed in Report 1 and
Report 2.
