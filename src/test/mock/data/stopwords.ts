export const stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"];

export const norm_bible = [
  'food used hunger',
  'food used eat',
  'need food eat',
  'need food hunger',
  'want food hunger',
  'want food eat',
  'want car drive',
  'need car drive',
  'need car travel',
  'want car travel',
  'use fly travel fast',
  'fast fly travel',
  'need food play',
  'need food live',
  'need eat live',
  'tesla kind car',
  'tesla fast car',
  'tesla is an electric car',
  'electric is fast',
  'car fly walk ways travel',
  'want tesla drive car',
  'want food solve hunger',
  'need eat food hunger',
  'want tesla drive fast far',
  'need plane fly'];

export const norm_bible_matrix = norm_bible.map(nb => nb.split(' ').filter(nb => !stop_words.includes(nb)));

export const products = [
  ['chair_1', 'chair_2', 'table_1', 'chair_1', 'chair_4',],
  ['chair_10', 'chair_9', 'chair_1', 'chair_5', 'chair_4',],
  ['chair_7', 'chair_6', 'chair_1', 'chair_3', 'chair_4',],
  ['chair_2', 'chair_1', 'table_4', 'table_2', 'chair_10',],
  ['chair_2', 'chair_1', 'table_4', 'table_2', 'chair_3',],
  ['chair_1', 'chair_2', 'chair_3', 'chair_4', 'chair_5',],
  ['chair_1', 'chair_10', 'chair_2', 'chair_8', 'chair_2','chair_1', 'chair_10', 'chair_2', 'chair_8', 'chair_2',],
  ['chair_9', 'chair_10', 'chair_8', 'chair_7', 'chair_6',],
  ['chair_8', 'chair_6', 'chair_3', 'chair_6', 'chair_5',],
  ['desk_1', 'chair_6', 'desk_2', 'table_1', 'desk_1',],
  ['desk_6', 'desk_5', 'desk_2', 'desk_1', 'table_1',],
  ['desk_2', 'desk_1', 'desk_2', 'desk_1', 'desk_2',],
  ['desk_1', 'desk_1', 'desk_2', 'desk_1', 'desk_2',],
  ['desk_2', 'desk_1', 'desk_2', 'desk_1', 'desk_2',],
  ['desk_2', 'desk_1', 'desk_2', 'desk_2', 'desk_2',],
  ['chair_6', 'chair_6', 'chair_2', 'chair_1', 'table_1',],
  ['desk_5', 'desk_6', 'desk_2', 'desk_1', 'table_1',],
  ['desk_2', 'desk_1', 'table_4', 'table_2', 'desk_10',],
  ['table_6', 'desk_6', 'table_2', 'table_1', 'table_10',],
  ['table_7', 'table_3', 'table_4', 'table_5',],
  ['table_5', 'table_2', 'table_3', 'table_4', 'table_6',],
  ['table_7', 'table_8', 'table_3', 'table_4', 'table_5',],
  ['table_9', 'table_8', 'table_6', 'table_4', 'table_5',],
  ['table_5', 'table_6', 'table_2', 'table_1', 'desk_1',],
];

// console.log(products.map(prodRow=>`${prodRow.join(' ')}.`))

export const furniture = [
  ['bigdresser_174028', 'bigdresser_178963', 'bigdresser_545842', 'bigdresser_545843', 'bigdresser_512954', 'bigdresser_589216', 'bigdresser_561757',],
  ['chair_2486686', 'chair_1952928', 'chair_2616975', 'chair_2568841', 'chair_1682450', 'chair_2518360',
    // 'chair_2510157',
    // 'chair_1733402',
    // 'chair_589216',
    // 'chair_2485252',
  ],
  // ['vent_2260883', 'vent_1019998', 'vent_1019980', 'vent_174899', 'vent_2088087', 'vent_2145379', 'vent_1061440', 'vent_1043549',],
  ['dresser_278102', 'dresser_278102', 'dresser_278103', 'dresser_174028', 'dresser_221836', 'dresser_281216', 'dresser_278102',
    // 'dresser_143036', 'dresser_278104', 'dresser_278104',
  ],
  // ['desk_526413', 'desk_521910', 'desk_481473', 'desk_288156', 'desk_363625', 'desk_281795', 'desk_243742', 'desk_283653', 'desk_223105', 'desk_185395',],
  // ['tabletset_533309', 'tabletset_545628', 'tabletset_534253', 'tabletset_534254', 'tabletset_509622', 'tabletset_537229', 'tabletset_517712', 'tabletset_545583', 'tabletset_533308', 'tabletset_192498',],
  // ['lighting_1042379', 'lighting_1076715', 'lighting_1044560', 'lighting_1044565', 'lighting_1076775', 'lighting_1041719', 'lighting_1041719', 'lighting_1036719', 'lighting_1583676', 'lighting_1026373', 'lighting_1076670',],
  // ['babyset_239060', 'babyset_239060', 'babyset_1764425', 'babyset_252263', 'babyset_238455', 'babyset_517184', 'babyset_526874', 'babyset_532777', 'babyset_515616',],
  // ['bedding_2236157', 'bedding_2503863', 'bedding_1883861', 'bedding_2673671', 'bedding_2563599', 'bedding_2131437', 'bedding_2122085', 'bedding_1932659', 'bedding_1119624', 'bedding_1928687',],
  // ['rug_1203924', 'rug_1203961', 'rug_1203952', 'rug_1059474', 'rug_1204109', 'rug_1203980', 'rug_1236740', 'rug_1204005', 'rug_1203949', 'rug_2617126', 'rug_1203961', 'rug_1059474', 'rug_1203952'],
];


const series = [
  {
    name:'PAD',
    data:[[ -5.2228196735970185, -24.699014366642842 ]],
  },
  {
    name:'chair_1',
    color:'red',
    data:[[ 40.48355499918796, 199.60251905626941 ]],
  },
  {
    name:'chair_2',
    color:'red',
    data:[[ -121.73011295452665, -133.96194532874128 ]],
  },
  {
    name:'table_1',
    color:'blue',
    data:[[ 15.731380400383951, 27.06089365395619 ]],
  },
  {
    name:'chair_4',
    color:'red',
    data:[[ 16.83651838500691, -31.087547578175055 ]],
  },
  {
    name:'chair_10',
    color:'red',
    data:[[ 29.88219341758057, -48.43939603139351 ]],
  },
  {
    name:'chair_9',
    color:'red',
    data:[[ 116.56993623207275, 117.1665232433465 ]],
  },
  {
    name:'chair_5',
    color:'red',
    data:[[ -16.814462835662308, 52.1615932376446 ]],
  },
  {
    name:'chair_7',
    color:'red',
    data:[[ -33.90063396831023, -6.14578686284678 ]],
  },
  {
    name:'chair_6',
    color:'red',
    data:[[ -0.7428586101751219, 13.313812340784244 ]],
  },
  {
    name:'chair_3',
    color:'red',
    data:[[ -27.60068481421351, -27.678686799340547 ]],
  },
  {
    name:'table_4',
    color:'blue',
    data:[[ -76.50997834402695, 34.96707659217015 ]],
  },
  {
    name:'table_2',
    color:'blue',
    data:[[ 49.13804885718071, -30.379200985267815 ]],
  },
  {
    name:'chair_8',
    color:'red',
    data:[[ 23.370624366031855, 45.02165383697269 ]],
  },
  {
    name:'desk_1',
    color:'green',
    data:[[ -24.607282481921654, 11.422781300643338 ]],
  },
  {
    name:'desk_2',
    color:'green',
    data:[[ 65.68127207939418, 9.76429377439894 ]],
  },
  {
    name:'desk_6',
    color:'green',
    data:[[ 26.91272604458441, 11.746706909555815 ]],
  },
  {
    name:'desk_5',
    color:'green',
    data:[[ -4.032426277219081, 33.894815334253906 ]],
  },
  {
    name:'desk_10',
    color:'green',
    data:[[ 49.49100320216197, -5.1660566592258395 ]],
  },
  {
    name:'table_6',
    color:'blue',
    data:[[ 4.714850094205763, -60.543865165066705 ]],
  },
  {
    name:'table_10',
    color:'blue',
    data:[[ -153.55667818629573, -121.18246639935901 ]],
  },
  {
    name:'table_7',
    color:'blue',
    data:[[ -10.439752504748203, -46.35397065286619 ]],
  },
  {
    name:'table_3',
    color:'blue',
    data:[[ -29.376320245225077, 31.2948246449049 ]],
  },
  {
    name:'table_5',
    color:'blue',
    data:[[ -11.036324179271627, -5.512363250476191 ]],
  },
  {
    name:'table_8',
    color:'blue',
    data:[[ 10.304478139294904, -7.406680775125614 ]],
  },
  {
    name:'table_9',
    color:'blue',
    data:[[ 29.847133692180453, -10.945642958436004 ]],
  }
]
