def test_public_imports():
    from Cebra_GW.stratified_gm import (
        CEBRAAnalysis,
        CEBRAUtils,
        run_analysis,
        split_point_clouds,
    )

    assert CEBRAAnalysis is not None
    assert CEBRAUtils is not None
    assert callable(run_analysis)
    assert callable(split_point_clouds)