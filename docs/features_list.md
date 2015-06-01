## Features


### Predicting MOOC Dropout over Weeks Using Machine Learning Methods [PDF](http://www2.informatik.hu-berlin.de/~kloftmar/publications/emnlp_mooc.pdf) [NOTE](http://meefen.github.io/notes/2014/09/11/kloft14predicting/)

| Name                     | Descriptions                                      |
|:-------------------------|:--------------------------------------------------|
| [] Number of requests    | Total number of requests including **page views** and **video click** actions |
| [] Number of sessions    | Number of sessions is supposed to be a reflection of high engagement, because more sessions indicate more often logging into the learning platform |
| [] Number of active days | We define a day as an active day if the student had at least one session on that day  |
| [] Number of page views  | The page views include lecture pages, wiki pages, homework pages and forum pages |
| [] Number of page views per session | The **average** number of pages viewed by each participant per session |
| [] Number of video views | Total number of video click actions |
| [] Number of video views per session | average number of video click actions per session |
| [] Number of forum views | Number of course discussion forum views |
| [] Number of wiki views  | number of course wiki page views |
| [] Number of homework page views | Times of access problem page |
| [U] Number of straight-through video plays | This is a video action attribute. Straight-trough playing video means that the participates played video without any jump (e.g. pause, resume, jump backward and jump forward). Since the lecture videos are the most important learning resource for the learning participants, the video playing should be investigated as other researchers did (Brotherton and Abowd, 2004). In this paper, five video behaviors are taken into account including the number of full plays as well as four others: start-stop during video plays, skip-ahead during video plays, relisten during video plays and the use of low play rate |
| [U] Number of start-stop during video plays | start-stop during video plays stands for a lecture video being paused and resumed |
| [U] Number of skip-ahead during video plays | skip-ahead means that the participant played a video with a forward jump |
| [U] Number of relisten during video plays | relisten means that a backward jump was made as the participant was playing a video |
| [U] Number of slow play rate use | this attribute is considered as an indicator of weak understanding of the lecturer’s lecture presentation, possibly because of language difficulties or a lack of relevant background knowledge |
| [] Most common request time | our attempt with this attribute is to separate day time learning from night time learning. We define night time from 19:00 to 6:59 in the morning and the other half day as day time |
| [] Number of requests from outside of Coursera | this is to discover how many requests from third-party tools (such as e-mail clients and social networks) to the course were made, which could be an indicator of the participant’s social behavior |
| [U] Number of screen pixels | the screen pixels is an indicator of the device that the student used. Typically, mobile devices come with fewer pixels |
| [] Most active day | through this attribute, we can investigate if starting late or early could have an impact on dropout |
| [U] Country: this information could reflect geographical differences in learning across the world |
| [U] Operating System | |
| [] Browser | |


### Identifying At-Risk Students in Massive Open Online Courses [PDF](http://www.ruizhang.info/publications/AAAI2015-MOOC.pdf)


### Predicting MOOC Performance with Week 1 Behavior [PDF](http://educationaldatamining.org/EDM2014/uploads/procs2014/short%20papers/273_EDM-2014-Short.pdf)

    - [] The first predictor is the average quiz score learners obtained in the first week of the course.
    - [] The second predictor is the number of peer assessments students completed in Week 1.
    - [] The third predictor is learners’ social network degree in the first week, which measures the level of social integration.
    - [] The fourth predictor is whether or not a learner is an incoming UCI Undeclared major student. This subgroup of students will go on to take the Bio 93 onsite course and have received external incentive to participate in the online course. We identified students as Undeclared by matching their school email addresses with Coursera accounts.


[?] [The number and frequency of forum posts][yang-2013]

[?] [linguistic and structural features from forum interactions][ramesh-2013]

[?] [**weekly** aggregated behaviour features][kloft-2014]

    - [?] number of video views
    - [?] number of active days
    - [x] non-behaviour features, such as country for analysis of dropout rates.

[?] [4 features][halawa-2014]
    - [x] video-skip
    - [?] assignment-skip
    - [?] lag
    - [x] assignment performance

[?] [temporal features][ramesh-2013]
    - lastQuiz
    - lastLecture

[?] [pre-deadline submission time][taylor-2014]

[?] [one week data][jiang-2014]

[?] [temporal features][hgt-2014]

- [?] [the dropout week (when a student watched less than 10% of the remaining lectures and stopped submitting assignments)][hgt-2014]

- [?] [the final grade (normal certificate or distinction) upon completing the course][hgt-2014]
        - [?] video lecture downloads
        - [?] taking weekly quizzes
        - [?] solving peer assessments


- [?] additional features:

        - [?] number of lecture views and video quiz attempts by week;
        - [?] temporal information such as when a lecture was viewed or an assessment started during the week.

    We performed a Pearson’s r correlation between individual features and the student’s final performance (normal completion vs. distinction) and found that our higher granularity features, e.g., the number of video quizzes taken per week and when a lecture video was first accessed, increased accuracy in predicting dropout and final performance over earlier studies. Similarly, the time when students started peer-graded assessments were a good early predictor of their dropout rate and performance. Once scores on a peer assessment were available, they became the best indicators of performance.

    Our initial analysis of week 1 data indicated two dominant groups: 1) students who watched lectures and took the assigned quizzes (1,699 students) and 2) students who only watched lectures (6,953 students). This grouping was a strong indicator of dropouts: 60% of the lecture-only group dropped out by week 4 whereas only 20% of the quiz takers dropped out. Further, the average grade for the lecture- only students was 3.2%. This number was 66% for the other students. Overall, analysis showed that more precise temporal features and more quantitative information improved early prediction accuracies and false alarm rates as compared to using only assessment score features.


- Miaomiao Wen, Diyi Yang and Carolyn P. Rose. Linguistic Reflections of Student Engagement in Massive Open Online Courses. ICWSM’14, 2014.
    * linguistic analysis of the MOOC forum data;
    * However, only few MOOC students (roughly 5-10%) use the discussion forums (Rose and Siemens, 2014), so that dropout predictors for the remaining 90% would be desirable.


- Carolyn Rose and George Siemens. Shared Task on Prediction of Dropout Over Time in Massively Open Online Courses. Proceedings of the 2014 Empirical Methods in Natural Language Processing Workshop on Modeling Large Scale Social Interaction in Massively Open Online Courses, Qatar, October 2014.


[yang-2013]: (http://lytics.stanford.edu/datadriveneducation/papers/yangetal.pdf)
[ramesh-2013]: (http://linqs.cs.umd.edu/basilic/web/Publications/2013/ramesh:nipsws13/ramesh-nipsws13.pdf)
[kloft-2014]: (http://www2.informatik.hu-berlin.de/~kloftmar/publications/emnlp_mooc.pd)
[halawa-2014]: (https://oerknowledgecloud.org/sites/oerknowledgecloud.org/files/In_depth_37_1%252520(1).pdf)
[taylor-2014]: (http://arxiv.org/pdf/1408.3382v1.pdf)
[jiang-2014]: (http://educationaldatamining.org/EDM2014/uploads/procs2014/shortpapers/273_EDM-2014-Short.pdf)
[hgt-2014]: (http://epress.lib.uts.edu.au/journals/index.php/JLA/article/download/4212/4429)
