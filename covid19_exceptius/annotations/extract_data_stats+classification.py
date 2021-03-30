import pandas as pd


def get_data(inputf):

    """
    :param inputf: csv file with annotated data
    :return: dataframe format for classification
    """

    data_df = pd.read_csv(inputf, sep=',', header=0)

    # select event for stats

    event_1_stats = data_df['Event 1'].value_counts().rename_axis('unique_values').to_frame('counts').reset_index()
    event_2_stats = data_df['Event 2'].value_counts().rename_axis('unique_values').to_frame('counts').reset_index()
    event_3_stats = data_df['Event 3'].value_counts().rename_axis('unique_values').to_frame('counts').reset_index()
    event_4_stats = data_df['Event 4'].value_counts().rename_axis('unique_values').to_frame('counts').reset_index()
    event_5_stats = data_df['Event 5'].value_counts().rename_axis('unique_values').to_frame('counts').reset_index()
    event_6_stats = data_df['Event 6'].value_counts().rename_axis('unique_values').to_frame('counts').reset_index()
    event_7_stats = data_df['Event 7'].value_counts().rename_axis('unique_values').to_frame('counts').reset_index()
    event_8_stats = data_df['Event 8'].value_counts().rename_axis('unique_values').to_frame('counts').reset_index()
    sanction_stats = data_df['Sanctions event'].value_counts().rename_axis('unique_values').to_frame('counts').reset_index()
    target_stats = data_df['Target group'].value_counts().rename_axis('unique_values').to_frame('counts').reset_index()


    # print event stats

    event_1_stats.to_csv('event1_stats.csv', index=False)
    event_2_stats.to_csv('event2_stats.csv', index=False)
    event_3_stats.to_csv('event3_stats.csv', index=False)
    event_4_stats.to_csv('event4_stats.csv', index=False)
    event_5_stats.to_csv('event5_stats.csv', index=False)
    event_6_stats.to_csv('event6_stats.csv', index=False)
    event_7_stats.to_csv('event7_stats.csv', index=False)
    event_8_stats.to_csv('event8_stats.csv', index=False)
    sanction_stats.to_csv('sanction_stats.csv', index=False)
    target_stats.to_csv('target_stats.csv', index=False)

    # dataset for event classification

    df_event_classification = data_df[['Paragraph text', 'Event 1', 'Event 2', 'Event 3', 'Event 4', 'Event 5', 'Event 6', 'Event 7', 'Event 8']].fillna(0)

    df_event_classification[['Event 1', 'Event 2', 'Event 3', 'Event 4', 'Event 5',
                             'Event 6', 'Event 7', 'Event 8']] = df_event_classification[
        ['Event 1', 'Event 2', 'Event 3', 'Event 4', 'Event 5',
         'Event 6', 'Event 7', 'Event 8']].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(1)

    #print(df_event_classification.head())

    df_event_classification[['Event 1', 'Event 2', 'Event 3', 'Event 4', 'Event 5',
                             'Event 6', 'Event 7', 'Event 8']] = df_event_classification[
        ['Event 1', 'Event 2', 'Event 3', 'Event 4', 'Event 5',
         'Event 6', 'Event 7', 'Event 8']].astype(int)


    df_event_classification['Classification_label'] = df_event_classification[df_event_classification.columns[1:]].apply(lambda x: ','.join(x.astype(str)),axis=1)

#    print(df_event_classification.head())

    event_classification_format_v1 = df_event_classification[['Paragraph text', 'Classification_label']]
    event_classification_format_v2 = df_event_classification[['Paragraph text', 'Event 1', 'Event 2', 'Event 3', 'Event 4', 'Event 5',
         'Event 6', 'Event 7', 'Event 8']]



    return event_classification_format_v1, event_classification_format_v2

if __name__ == '__main__':

    inputf = 'uk_annotations_full_1k.csv'

    dat4classifier_v1, dat4classifier_v2 = get_data(inputf)

    dat4classifier_v1.to_csv('event_classes_multilabel_v1.csv', index=False)
    dat4classifier_v2.to_csv('event_classes_multilabel_v2.csv', index=False)



