
// Returns a random number between min (inclusive) and max (exclusive)
function getRandomArbitrary(min, max) {
    return Math.random() * (max - min) + min;
}

// first, produce a line chart for metrics wrt model runs
var datasets = [];
var labels = [];
for(var i = 0; i < WANDB_RUN[0]['history'].length; i++) {
    labels.push(WANDB_RUN[0]['history'][i]['epoch']);
}
for(var i = 0; i < WANDB_RUN.length; i++) {
    var h = WANDB_RUN[i]['history'];
    var data = [];
    for (var j = 0; j < h.length; j++) {
        data.push(h[j]['loss'])
    }
    var color = 'rgb(' + getRandomArbitrary(0, 256) + ', '
        + getRandomArbitrary(0, 256) + ', ' + getRandomArbitrary(0, 256) + ')';
    var d = {
                label: WANDB_RUN[i].name,
                fill: false,
                borderColor: color,
                backgroundColor: color,
                data: data
            };
    datasets.push(d);
}
// FROM: https://www.chartjs.org/docs/latest/getting-started/
var ctx = document.getElementById('modelCanvas').getContext('2d');
var chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: labels,
        datasets: datasets
    },
    // Label the axes
    options: {
        scales: {
            yAxes: [{
                scaleLabel: {
                    display: true,
                    labelString: 'loss'
                }
            }],
            xAxes: [{
                scaleLabel: {
                    display: true,
                    labelString: 'epoch'
                }
            }]
        }
    }
});
console.log('Chart done!');

// second script, bar chart for the distributions of runs initiated by the users
var ctx_bars = document.getElementById('ownerCanvas').getContext('2d');
var user_labels = Object.keys(USER_RUNS);
var user_data = [];
for(var i = 0; i < user_labels.length; i ++) {
    user_data.push(USER_RUNS[user_labels[i]]);
}
var chart_bars = new Chart(ctx_bars,
    {
        "type":"bar",
        "data":{
            "labels":  user_labels,
            "datasets":[
                {
                    "label":"Runs per user",
                    "data": user_data,
                    "fill": false,
                    "backgroundColor": "rgba(65, 105, 225, 0.2)",
                    "borderColor": "rgb(65, 105, 225)",
                    "borderWidth":1
                }
                ]
        },
        "options": {
            "scales":
                {"yAxes":
                        [
                            {
                                "ticks": {"beginAtZero": true},
                                scaleLabel: { labelString: 'runs'}
                            }
                            ]
                    }
            }
    });