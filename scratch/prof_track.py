import cProfile, pstats, torch, sys
from yolo_kalman.pipeline import Pipeline, PipelineConfig
sys.path.insert(0, "scripts")
clips = ["clips/MD_355_and_MD_124_Gaithersburg_MD_2022_07_30_11_07_03_1.webm","clips/Soundscapes_of_Norrk_ping__1____Skvallertorget.webm","clips/Soundscapes_of_Norrk_ping__4____Train_station.webm","clips/Starling_bathing_one_sparrow_eating_two_doves_walking_around.webm","clips/Traffic_on_Slovenska_street.webm"]*2
cfg = PipelineConfig(max_frames=600, overlap=False, classes=[0,1,2,3,5,7,14,15,16])
p = Pipeline(clips[:8], cfg=cfg)
pr = cProfile.Profile(); pr.enable(); p.run(); pr.disable()
print(p.wall, p.timing, p.tracker.timing)
st = pstats.Stats(pr); st.sort_stats("tottime").print_stats(25)
