
ethnicities = {
    'white': ['White - English/Welsh/Scottish/Northern Irish/British',
              'White - Any other White background',
              'White - Irish'
              'White - Gypsy or Irish Traveller'
              ],
    'black': ['Black/African/Caribbean/Black British - African',
              'Black/African/Caribbean/Black British - Caribbean',
              'Black/African/Caribbean/Black British - Any other Black/African/Caribbean background'],
    'asian': [
        'Asian/Asian British - Pakistani',
        'Asian/Asian British - Indian',
        'Asian/Asian British - Bangladeshi',
        'Asian/Asian British - Any other Asian background',
        'Asian/Asian British - Chinese']
    ,
    'mixed': [
        'Mixed/Multiple ethnic groups - White and Black African',
        'Mixed/Multiple ethnic groups - White and Black Caribbean',
        'Mixed/Multiple ethnic groups - Any other Mixed/Multiple ethnic background',
        'Mixed/Multiple ethnic groups - White and Asian']
    ,
    'other': [
        'Other ethnic group - Any other ethnic group',
        'Other ethnic group - Arab'
    ],
    'missing': ['Other ethnic group - Not stated']
}


known_categories = {
    #"Officer-defined ethnicity": {"values": ["white", "black", "asian", "mixed", "other"], "default": "white"},
    #"Gender": {"values": ["male", "female"], "default": "male"},
    #"Age range": {"values": ['18-24', '25-34', 'over 34', '10-17', 'under 10'], "default": "over 34"},
    "Type": {"values": ['person search', 'person and vehicle search', 'vehicle search'], "default": "person search"},
    "Part of a policing operation": {"values": [True, False], "default": "missing"},
    "Legislation": {"values": ['misuse of drugs act 1971 (section 23)',
                               'police and criminal evidence act 1984 (section 1)',
                               'psychoactive substances act 2016 (s36(2))',
                               'criminal justice act 1988 (section 139b)',
                               'firearms act 1968 (section 47)',
                               'poaching prevention act 1862 (section 2)',
                               'criminal justice and public order act 1994 (section 60)',
                               'police and criminal evidence act 1984 (section 6)',
                               'wildlife and countryside act 1981 (section 19)',
                               'psychoactive substances act 2016 (s37(2))',
                               'aviation security act 1982 (section 27(1))',
                               'protection of badgers act 1992 (section 11)',
                               'crossbows act 1987 (section 4)',
                               'public stores act 1875 (section 6)',
                               'customs and excise management act 1979 (section 163)',
                               'deer act 1991 (section 12)',
                               'conservation of seals act 1970 (section 4)'],
                    "default": 'misuse of drugs act 1971 (section 23)'},
    "Object of search": {"values": ['controlled drugs', 'offensive weapons', 'stolen goods',
                                    'article for use in theft', 'articles for use in criminal damage',
                                    'firearms', 'anything to threaten or harm anyone', 'crossbows',
                                    'evidence of offences under the act', 'fireworks',
                                    'psychoactive substances', 'game or poaching equipment',
                                    'evidence of wildlife offences',
                                    'detailed object of search unavailable',
                                    'goods on which duty has not been paid etc.',
                                    'seals or hunting equipment'], "default": 'controlled drugs'},
    #"station": {"values": ['devon-and-cornwall', 'dyfed-powys', 'derbyshire', 'bedfordshire',
                           # 'avon-and-somerset', 'cheshire', 'sussex', 'north-yorkshire',
                           # 'cleveland', 'merseyside', 'north-wales', 'wiltshire', 'norfolk',
                           # 'suffolk', 'thames-valley', 'durham', 'warwickshire',
                           # 'leicestershire', 'hertfordshire', 'cumbria', 'essex',
                           # 'south-yorkshire', 'surrey', 'staffordshire', 'northamptonshire',
                           # 'northumbria', 'city-of-london', 'nottinghamshire',
                           # 'gloucestershire', 'cambridgeshire', 'lincolnshire', 'btp',
                           # 'west-yorkshire', 'dorset', 'west-mercia', 'kent', 'hampshire',
                           # 'humberside', 'lancashire', 'greater-manchester', 'gwent'], "default": "merseyside"}
}


outcomes_true = {'Local resolution',
                 'Community resolution',
                 'Offender given drugs possession warning',
                 'Khat or Cannabis warning',
                 'Caution (simple or conditional)',
                 'Offender given penalty notice',
                 'Arrest',
                 'Penalty Notice for Disorder',
                 'Suspected psychoactive substances seized - No further action',
                 'Summons / charged by post',
                 'Article found - Detailed outcome unavailable',
                 'Offender cautioned',
                 'Suspect arrested',
                 'Suspect summonsed to court'}
