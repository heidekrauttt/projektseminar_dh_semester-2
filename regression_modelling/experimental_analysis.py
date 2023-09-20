import pandas as pd
import plotly.express as plotly

# Read the CSV file and convert it to a pandas dataframe
df = pd.read_csv('../output_mit_humor.csv', delimiter=',')

# Group the dataframe by 'Generated Filename' and count the occurrences in each trial
summary_df = df.groupby('generated_filenames')['trial'].unique().reset_index()
summary_df['trial_count'] = summary_df['trial'].apply(len)

# sort it
summary_df = summary_df.sort_values('trial_count', ascending=False)

# Plot 1: the summary of duplicate entries
# fig = plotly.bar(summary_df, x='generated_filenames', y='trial_count', title='Duplicate Entries Summary')
# fig.show()

# Plot 2: 'trial' and 'number_of_outcomes' using Plotly
fig2 = plotly.line(df, x='trial', y='num_generations', title='Number of Outcomes by Trial', color='Occurrences')
fig2.show()

# Plot 3
# columns_to_plot = df.columns.difference(['trial', 'generated_filenames', 'trial_count', 'num_of_outcomes'])
# #
# # # Plot the values of all remaining columns
# fig3 = plotly.scatter(df, x='trial', y=columns_to_plot, title='Values by Trial',
#                   color_discrete_sequence=plotly.colors.qualitative.Plotly, symbol_sequence=['cross', 'square'])
# fig3.show()
